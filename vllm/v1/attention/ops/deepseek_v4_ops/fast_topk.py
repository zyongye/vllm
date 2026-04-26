# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 indexer top-k (k = 512), ported from sglang's topk_v2 family.

The kernel is registered as ``torch.ops._C.fast_topk_v2`` (selection +
fused page-table gather) and ``torch.ops._C.fast_topk_v2_plan`` (per-batch
threshold and metadata). It is built for Hopper (sm_90a) and Blackwell
datacenter (sm_100/sm_103); sm_120 (consumer Blackwell) is not supported
because it lacks thread-block clusters.

Two-step usage::

    metadata = plan_topk_v2(seq_lens)
    page_indices = fast_topk_v2(scores, seq_lens, page_table, page_size,
                                metadata=metadata)

The plan can be amortized across same-shape forward calls (e.g. across all
indexer layers within one cudagraph capture), since it depends only on
``seq_lens``.
"""

from typing import Optional

import torch

# The metadata layout (rows of int32x4) is fixed by the kernel; the planner
# writes one GlobalMetadata row + one row per batch entry.
_PLAN_COLS = 4

# Output top-k size. Hardcoded in the kernel.
_TOPK = 512


def workspace_ints_per_batch() -> int:
    """Number of int32s the kernel needs in `(B, _)` workspace per row."""
    return int(torch.ops._C.fast_topk_v2_workspace_ints())


def plan_topk_v2(
    seq_lens: torch.Tensor,
    static_cluster_threshold: int = 0,
    metadata: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pick a per-batch ``cluster_threshold`` and build the per-row metadata.

    Args:
        seq_lens: int32 tensor of shape ``(B,)``. CUDA, contiguous. Each entry
            is the number of valid scored tokens for that row (i.e., the
            indexer's compressed seq_len).
        static_cluster_threshold: when nonzero, override the auto-tuned
            threshold and route any row with ``seq_len > threshold`` through
            the cluster path. 0 means auto.
        metadata: optional preallocated int32 tensor of shape ``(B + 1, 4)``.
            When provided, the planner writes into it (useful for cudagraph
            capture). When omitted, a fresh tensor is allocated.

    Returns:
        The metadata tensor.
    """
    assert seq_lens.dim() == 1
    assert seq_lens.dtype == torch.int32
    assert seq_lens.is_cuda and seq_lens.is_contiguous()

    batch_size = seq_lens.size(0)
    if metadata is None:
        metadata = torch.empty(
            (batch_size + 1, _PLAN_COLS),
            dtype=torch.int32,
            device=seq_lens.device,
        )
    else:
        assert metadata.shape == (batch_size + 1, _PLAN_COLS)
        assert metadata.dtype == torch.int32
        assert metadata.is_cuda and metadata.is_contiguous()

    torch.ops._C.fast_topk_v2_plan(
        seq_lens, metadata, int(static_cluster_threshold)
    )
    return metadata


def fast_topk_v2(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    *,
    metadata: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    page_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Select top-512 indexer scores per row and fold the page-table gather.

    Args:
        scores: float32 logits of shape ``(B, L)``. ``stride(-1) == 1`` and
            ``stride(0) % 4 == 0`` (TMA 16-byte alignment).
        seq_lens: int32 ``(B,)``; only the first ``seq_lens[b]`` columns are
            considered for row b.
        page_table: int32 ``(B, max_blocks)`` with ``stride(-1) == 1``.
        page_size: power-of-2 page size used by the indexer KV cache.
        metadata: optional plan tensor from :func:`plan_topk_v2`. If omitted,
            it is built on the fly.
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

    if page_indices is None:
        page_indices = scores.new_empty(
            (batch_size, _TOPK), dtype=torch.int32
        )
    if workspace is None:
        workspace = scores.new_empty(
            (batch_size, workspace_ints_per_batch()), dtype=torch.int32
        )
    if metadata is None:
        metadata = plan_topk_v2(seq_lens)

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
