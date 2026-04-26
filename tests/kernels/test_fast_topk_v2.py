# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for fast_topk_v2 (DeepSeek V4 indexer top-k, k=512).

Run::

    .venv/bin/python -m pytest tests/kernels/test_fast_topk_v2.py -v

Coverage:
- All four execution paths: trivial (sl<=512), Register (1- and 2-pass),
  Streaming, and Cluster.
- Both launch shapes: fused (batch<=kNumClusters=15) and two-stage (>15).
- Mixed-length batches that exercise the per-row dispatch in the stage-2
  combine kernel.
- Page-table fold-in: parametrised across page_size in {1, 32, 64}.

The kernel emits page-table-resolved indices. By using
``page_table[b, i] = i`` with ``page_size=1`` we can compare the kernel's
output 1:1 against ``torch.topk`` on the masked scores. For other page sizes
the test inverts the page resolution before comparing.
"""

from __future__ import annotations

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.deepseek_v4_ops.fast_topk import (
    fast_topk_v2,
    plan_topk_v2,
    workspace_ints_per_batch,
)

# Match the kernel's compile-time constant.
TOPK = 512

# Thresholds inside the kernel (mirrors values in topk/register.cuh,
# topk_v2.cuh). Keep these in sync if the kernel changes.
SMALL_1PASS = 4 * 4 * 1024            # RegisterTopK::kMax1PassLength
SMALL_2PASS = 2 * SMALL_1PASS         # RegisterTopK::kMax2PassLength = 32768
DEFAULT_CLUSTER_THRESHOLD = SMALL_2PASS  # plan picks this for batch<=30
NUM_CLUSTERS = 15                     # kNumClusters in fast_topk_v2.cu


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _max_blocks_for(seq_len: int, page_size: int) -> int:
    return (seq_len + page_size - 1) // page_size


def _trivial_page_table(batch_size: int, max_blocks: int,
                        device: torch.device) -> torch.Tensor:
    """Identity page table: page_table[b, i] = i, so page_to_indices is a no-op
    when ``page_size == 1`` (page_bits == 0)."""
    return (
        torch.arange(max_blocks, dtype=torch.int32, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
    )


def _shuffled_page_table(batch_size: int, max_blocks: int, seed: int,
                         device: torch.device) -> torch.Tensor:
    """Per-row independent permutation of [0, max_blocks)."""
    g = torch.Generator(device=device).manual_seed(seed)
    rows = []
    for _ in range(batch_size):
        rows.append(torch.randperm(max_blocks, generator=g, device=device,
                                   dtype=torch.int32))
    return torch.stack(rows, dim=0)


def _resolve(raw_idx: int, b: int, page_table: torch.Tensor,
             page_size: int) -> int:
    """Mirror of the device-side page_to_indices."""
    block = raw_idx // page_size
    offset = raw_idx % page_size
    return int(page_table[b, block]) * page_size + offset


def _invert_resolved(resolved_idx: int, b: int, page_table: torch.Tensor,
                     page_size: int) -> int:
    """Find a raw_idx in [0, max_blocks*page_size) such that
    _resolve(raw_idx, b) == resolved_idx. Used to translate kernel output
    back to raw scores for comparison with torch.topk."""
    block = resolved_idx // page_size
    offset = resolved_idx % page_size
    # Find the row in page_table[b] that holds `block`.
    matches = (page_table[b] == block).nonzero(as_tuple=False)
    assert matches.numel() == 1, (
        f"page_table row {b} is not a permutation: block {block} appears "
        f"{matches.numel()} times")
    return int(matches.item()) * page_size + offset


def _reference_topk(scores: torch.Tensor, seq_lens: torch.Tensor,
                    page_table: torch.Tensor, page_size: int) -> list[set[int]]:
    """Per-row reference: page-resolved set of indices that fast_topk_v2
    should emit (excluding -1 padding)."""
    B, _ = scores.shape
    out: list[set[int]] = []
    for b in range(B):
        sl = int(seq_lens[b])
        if sl <= TOPK:
            valid = list(range(sl))
        else:
            row = scores[b, :sl]
            _, raw = torch.topk(row, TOPK)
            valid = raw.tolist()
        out.append({_resolve(i, b, page_table, page_size) for i in valid})
    return out


def _check(scores: torch.Tensor, seq_lens: torch.Tensor,
           page_table: torch.Tensor, page_size: int) -> None:
    metadata = plan_topk_v2(seq_lens)
    workspace = scores.new_empty(
        (scores.shape[0], workspace_ints_per_batch()), dtype=torch.int32)
    indices = fast_topk_v2(scores, seq_lens, page_table, page_size,
                           metadata=metadata, workspace=workspace)
    torch.cuda.synchronize()

    expected = _reference_topk(scores, seq_lens, page_table, page_size)
    B = scores.shape[0]
    for b in range(B):
        sl = int(seq_lens[b])
        valid_count = min(sl, TOPK)
        row = indices[b].tolist()
        # Padding region: -1 (only when sl < TOPK).
        if sl < TOPK:
            assert all(v == -1 for v in row[sl:]), (
                f"row {b}: expected -1 padding after position {sl}, got "
                f"{row[sl:sl + 8]}")
        got = set(row[:valid_count])
        assert -1 not in got, f"row {b}: -1 inside valid region (sl={sl})"
        assert got == expected[b], (
            f"row {b} (sl={sl}, page_size={page_size}): "
            f"missing={len(expected[b] - got)} extra={len(got - expected[b])}")


# --------------------------------------------------------------------------
# Skip non-CUDA / non-Hopper-or-later
# --------------------------------------------------------------------------


def _supports_clusters() -> bool:
    if not current_platform.is_cuda():
        return False
    major, _ = torch.cuda.get_device_capability()
    # Thread-block clusters / TMA / PDL are sm_90+. sm_120 (consumer
    # Blackwell) is missing some of these; skip when we detect it.
    return major == 9 or major == 10


pytestmark = pytest.mark.skipif(
    not _supports_clusters(),
    reason="fast_topk_v2 requires sm_90 (Hopper) or sm_100 (Blackwell DC)",
)


# --------------------------------------------------------------------------
# Path coverage
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seq_lens", [
    pytest.param([1], id="trivial_1"),
    pytest.param([300], id="trivial_300"),
    pytest.param([512], id="trivial_boundary_512"),
    pytest.param([513, 600, 100, 511, 512], id="trivial_mix"),
])
def test_trivial_path(seq_lens):
    """sl <= 512: identity-style fill, no radix, no tie-break."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    B = len(seq_lens)
    L = max(max(seq_lens), 1024)            # round up so stride is multiple of 4
    L = (L + 3) & ~3
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    scores = torch.randn(B, L, dtype=torch.float32, device=device)
    page_table = _trivial_page_table(B, _max_blocks_for(L, 1), device)
    _check(scores, seq_lens_t, page_table, page_size=1)


@pytest.mark.parametrize("seq_len", [
    pytest.param(513, id="just_above_topk"),
    pytest.param(2048, id="2k"),
    pytest.param(SMALL_1PASS - 1, id="register_1pass_max"),
    pytest.param(SMALL_1PASS, id="register_1pass_boundary"),
    pytest.param(SMALL_1PASS + 1, id="register_2pass_first"),
    pytest.param(SMALL_2PASS - 1, id="register_2pass_max"),
])
def test_register_path(seq_len):
    """Register strategy (small N; both 1- and 2-pass)."""
    torch.manual_seed(seq_len)
    device = torch.device("cuda")
    B = 4
    L = (seq_len + 3) & ~3
    scores = torch.randn(B, L, dtype=torch.float32, device=device)
    seq_lens = torch.full((B,), seq_len, dtype=torch.int32, device=device)
    page_table = _trivial_page_table(B, _max_blocks_for(L, 1), device)
    _check(scores, seq_lens, page_table, page_size=1)


@pytest.mark.parametrize("seq_len", [
    pytest.param(SMALL_2PASS, id="streaming_first"),
    pytest.param(40000, id="streaming_40k"),
    pytest.param(DEFAULT_CLUSTER_THRESHOLD, id="streaming_at_cluster_thresh"),
])
def test_streaming_path(seq_len):
    """Streaming strategy (medium N). With small batch and seq_len <=
    auto-picked cluster_threshold (>= 32K when batch <= 30), the per-row
    dispatch routes here."""
    torch.manual_seed(seq_len)
    device = torch.device("cuda")
    B = 4
    L = (seq_len + 3) & ~3
    scores = torch.randn(B, L, dtype=torch.float32, device=device)
    seq_lens = torch.full((B,), seq_len, dtype=torch.int32, device=device)
    page_table = _trivial_page_table(B, _max_blocks_for(L, 1), device)
    _check(scores, seq_lens, page_table, page_size=1)


@pytest.mark.parametrize("batch_size,seq_len", [
    pytest.param(2, 65536, id="cluster_fused_64k"),
    pytest.param(NUM_CLUSTERS, 131072, id="cluster_fused_max_batch"),
    pytest.param(NUM_CLUSTERS + 1, 65536, id="cluster_two_stage_just_over"),
    pytest.param(32, 96000, id="cluster_two_stage_32x96k"),
])
def test_cluster_path(batch_size, seq_len):
    """Large strategy (Hopper thread-block clusters). Force seq_len above
    the auto threshold by passing static_cluster_threshold=SMALL_2PASS."""
    torch.manual_seed(seq_len * batch_size)
    device = torch.device("cuda")
    L = (seq_len + 3) & ~3
    scores = torch.randn(batch_size, L, dtype=torch.float32, device=device)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32,
                          device=device)
    page_table = _trivial_page_table(batch_size, _max_blocks_for(L, 1), device)
    metadata = plan_topk_v2(seq_lens, static_cluster_threshold=SMALL_2PASS)
    indices = fast_topk_v2(scores, seq_lens, page_table, page_size=1,
                           metadata=metadata)
    torch.cuda.synchronize()
    expected = _reference_topk(scores, seq_lens, page_table, page_size=1)
    for b in range(batch_size):
        got = set(indices[b].tolist())
        assert got == expected[b], (
            f"row {b}: missing={len(expected[b] - got)} "
            f"extra={len(got - expected[b])}")


@pytest.mark.parametrize("page_size", [1, 32, 64])
def test_page_table_fold_in(page_size):
    """page_to_indices: kernel-side fold of the page-table gather."""
    torch.manual_seed(page_size)
    device = torch.device("cuda")
    B, seq_len = 4, 6000
    L = (seq_len + 3) & ~3
    max_blocks = (L + page_size - 1) // page_size
    scores = torch.randn(B, L, dtype=torch.float32, device=device)
    seq_lens = torch.full((B,), seq_len, dtype=torch.int32, device=device)
    page_table = _shuffled_page_table(B, max_blocks, seed=page_size,
                                      device=device)
    _check(scores, seq_lens, page_table, page_size=page_size)


def test_mixed_lengths_route_per_row():
    """Per-row dispatch in topk_combine_transform: trivial / Register /
    Streaming / Cluster all in one batch. Use static_cluster_threshold to
    force a mix that includes the Large path."""
    torch.manual_seed(7)
    device = torch.device("cuda")
    seq_lens = [
        100,                 # trivial
        SMALL_1PASS - 100,   # 1-pass register
        SMALL_2PASS - 100,   # 2-pass register
        50000,               # streaming
        40000,               # streaming
        80000,               # cluster (above static_cluster_threshold)
    ]
    B = len(seq_lens)
    L = (max(seq_lens) + 3) & ~3
    scores = torch.randn(B, L, dtype=torch.float32, device=device)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    page_table = _trivial_page_table(B, _max_blocks_for(L, 1), device)

    # Force seq_len > 49152 to take the Cluster path.
    metadata = plan_topk_v2(seq_lens_t, static_cluster_threshold=49152)
    indices = fast_topk_v2(scores, seq_lens_t, page_table, page_size=1,
                           metadata=metadata)
    torch.cuda.synchronize()

    expected = _reference_topk(scores, seq_lens_t, page_table, page_size=1)
    for b, sl in enumerate(seq_lens):
        valid = min(sl, TOPK)
        row = indices[b].tolist()
        if sl < TOPK:
            assert all(v == -1 for v in row[sl:])
        got = set(row[:valid])
        assert got == expected[b], f"row {b} (sl={sl}) mismatched"


def test_metadata_can_be_reused_across_calls():
    """plan_topk_v2 is amortizable: same metadata reused across calls."""
    torch.manual_seed(123)
    device = torch.device("cuda")
    B, seq_len = 8, 4096
    L = (seq_len + 3) & ~3
    seq_lens = torch.full((B,), seq_len, dtype=torch.int32, device=device)
    page_table = _trivial_page_table(B, _max_blocks_for(L, 1), device)
    metadata = plan_topk_v2(seq_lens)

    # Two independent score buffers, same metadata.
    scores_a = torch.randn(B, L, dtype=torch.float32, device=device)
    scores_b = torch.randn(B, L, dtype=torch.float32, device=device)
    out_a = fast_topk_v2(scores_a, seq_lens, page_table, page_size=1,
                         metadata=metadata)
    out_b = fast_topk_v2(scores_b, seq_lens, page_table, page_size=1,
                         metadata=metadata)
    torch.cuda.synchronize()

    expected_a = _reference_topk(scores_a, seq_lens, page_table, page_size=1)
    expected_b = _reference_topk(scores_b, seq_lens, page_table, page_size=1)
    for b in range(B):
        assert set(out_a[b].tolist()) == expected_a[b]
        assert set(out_b[b].tolist()) == expected_b[b]


def test_workspace_can_be_preallocated():
    """Workspace passed in by the caller (cudagraph-friendly path)."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    B, seq_len = 16, 70000
    L = (seq_len + 3) & ~3
    seq_lens = torch.full((B,), seq_len, dtype=torch.int32, device=device)
    page_table = _trivial_page_table(B, _max_blocks_for(L, 1), device)
    scores = torch.randn(B, L, dtype=torch.float32, device=device)

    metadata = plan_topk_v2(seq_lens, static_cluster_threshold=SMALL_2PASS)
    workspace = scores.new_empty((B, workspace_ints_per_batch()),
                                 dtype=torch.int32)
    page_indices = scores.new_empty((B, TOPK), dtype=torch.int32)
    out = fast_topk_v2(scores, seq_lens, page_table, page_size=1,
                       metadata=metadata, workspace=workspace,
                       page_indices=page_indices)
    torch.cuda.synchronize()
    assert out.data_ptr() == page_indices.data_ptr(), (
        "kernel must write into the caller-supplied page_indices tensor")
    expected = _reference_topk(scores, seq_lens, page_table, page_size=1)
    for b in range(B):
        assert set(out[b].tolist()) == expected[b]
