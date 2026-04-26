# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 indexer top-k (k = 512), ported from sglang's topk_v2 family.

Three Python wrappers, all over the same underlying kernel:

- :func:`plan_topk_v2`     - run the plan kernel; produces per-batch metadata
                              that the main kernel reads.
- :func:`fast_topk_v2_raw` - select the top-512 columns per row, emit raw
                              row-local indices (drop-in for persistent_topk).
- :func:`fast_topk_v2`     - same selection + fold the page-table gather
                              into the kernel's output store.

Each main-kernel wrapper accepts an optional pre-allocated ``metadata``;
when omitted, ``plan_topk_v2`` is invoked internally with a fresh tensor.
Callers that want to amortize the plan kernel across multiple selections
(e.g. across all indexer layers in a single forward) should plan once and
pass the resulting metadata into every subsequent call.

Built for Hopper (sm_90a) and Blackwell datacenter (sm_100/sm_103); sm_120
(consumer Blackwell) is not supported because it lacks thread-block
clusters.
"""

import torch

# The metadata layout (rows of int32x4) is fixed by the kernel; the planner
# writes one GlobalMetadata row + one row per batch entry.
_PLAN_COLS = 4

# Output top-k size. Hardcoded in the kernel.
_TOPK = 512

_WORKSPACE_INTS_PER_BATCH: int | None = None


def workspace_ints_per_batch() -> int:
    """Number of int32s the kernel needs in ``(B, _)`` workspace per row.
    Cached after the first call; the value is a kernel-wide constant."""
    global _WORKSPACE_INTS_PER_BATCH
    if _WORKSPACE_INTS_PER_BATCH is None:
        _WORKSPACE_INTS_PER_BATCH = int(
            torch.ops._C.fast_topk_v2_workspace_ints()
        )
    return _WORKSPACE_INTS_PER_BATCH


def plan_topk_v2(
    seq_lens: torch.Tensor,
    static_cluster_threshold: int = 0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pick a per-batch ``cluster_threshold`` and build the per-row metadata.

    Args:
        seq_lens: int32 tensor of shape ``(B,)``. CUDA, contiguous. Each entry
            is the number of valid scored tokens for that row (i.e., the
            indexer's compressed seq_len).
        static_cluster_threshold: when nonzero, override the auto-tuned
            threshold and route any row with ``seq_len > threshold`` through
            the cluster path. 0 means auto.
        out: optional preallocated int32 tensor of shape ``(B + 1, 4)``.
            When provided, the planner writes into it (useful for cudagraph
            capture). When omitted, a fresh tensor is allocated.

    Returns:
        The metadata tensor.
    """
    assert seq_lens.dim() == 1
    assert seq_lens.dtype == torch.int32
    assert seq_lens.is_cuda and seq_lens.is_contiguous()

    batch_size = seq_lens.size(0)
    if out is None:
        out = torch.empty(
            batch_size + 1, _PLAN_COLS, dtype=torch.int32, device=seq_lens.device)

    torch.ops._C.fast_topk_v2_plan(
        seq_lens, out, int(static_cluster_threshold)
    )
    return out


def fast_topk_v2_raw(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    metadata: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    topk_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Top-k only: select the top-512 indices per row, no page-table fold-in.

    Drop-in replacement for ``torch.ops._C.persistent_topk``: emits raw
    row-local indices into ``topk_indices``. Use this when the caller wants
    to apply its own page-table translation later (or doesn't need one).
    The page-table loads inside the kernel are eliminated at compile time
    via ``if constexpr (kRawOutput)``.

    Args:
        scores: float32 ``(B, L)``, ``stride(-1)==1``, ``stride(0) % 4 == 0``.
        seq_lens: int32 ``(B,)``.
        metadata: optional preallocated int32 tensor of shape ``(B + 1, 4)``.
            When provided, the planner writes into it (useful for cudagraph
            capture). When omitted, a fresh tensor is allocated.
        workspace: optional preallocated ``(B, workspace_ints_per_batch())``
            int32, ``stride(-1) == 1``.
        topk_indices: optional preallocated output ``(B, 512)`` int32
            contiguous.

    Returns:
        ``(B, 512)`` int32 tensor of raw indices into ``scores[b, :]``,
        with ``-1`` padding when ``seq_lens[b] < 512``.
    """
    assert scores.dim() == 2
    assert scores.dtype == torch.float32
    assert scores.is_cuda
    batch_size = scores.size(0)

    if metadata is None:
        metadata = plan_topk_v2(seq_lens)

    if topk_indices is None:
        topk_indices = scores.new_empty(
            (batch_size, _TOPK), dtype=torch.int32
        )
    if workspace is None:
        workspace = scores.new_empty(
            (batch_size, workspace_ints_per_batch()), dtype=torch.int32
        )

    torch.ops._C.fast_topk_v2_raw(
        scores, seq_lens, topk_indices, workspace, metadata,
    )
    return topk_indices


def fast_topk_v2(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    *,
    metadata: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    page_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select top-512 indexer scores per row and fold the page-table gather.

    Args:
        scores: float32 logits of shape ``(B, L)``. ``stride(-1) == 1`` and
            ``stride(0) % 4 == 0`` (TMA 16-byte alignment).
        seq_lens: int32 ``(B,)``; only the first ``seq_lens[b]`` columns are
            considered for row b.
        page_table: int32 ``(B, max_blocks)`` with ``stride(-1) == 1``.
        page_size: power-of-2 page size used by the indexer KV cache.
        metadata: optional preallocated int32 tensor of shape ``(B + 1, 4)``.
            When provided, the planner writes into it (useful for cudagraph
            capture). When omitted, a fresh tensor is allocated.
        workspace: optional preallocated workspace ``(B, workspace_ints_per_batch())``
            int32, ``stride(-1) == 1``. Used for inter-cluster tie staging in
            the large-N path. Allocated on demand if absent.
        page_indices: optional preallocated output ``(B, 512)`` int32
            contiguous. Allocated on demand if absent.

    Returns:
        ``(B, 512)`` int32 tensor of page-table-resolved indices.
    """
    assert scores.dim() == 2
    assert scores.dtype == torch.float32
    assert scores.is_cuda
    batch_size = scores.size(0)

    if metadata is None:
        metadata = plan_topk_v2(seq_lens)
    else:
        metadata = metadata[: batch_size + 1]

    if page_indices is None:
        page_indices = scores.new_empty(
            (batch_size, _TOPK), dtype=torch.int32
        )
    if workspace is None:
        workspace = scores.new_empty(
            (batch_size, workspace_ints_per_batch()), dtype=torch.int32
        )
    torch.ops._C.fast_topk_v2(
        scores,
        seq_lens,
        page_table,
        page_indices,
        int(page_size),
        workspace,
        metadata,
    )
    return page_indices
