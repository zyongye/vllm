# SPDX-License-Identifier: Apache-2.0
import argparse
from types import SimpleNamespace

import torch
import torch.utils.benchmark as benchmark

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe
from vllm.model_executor.layers.fused_moe.triton_kernels_moe import (
    triton_kernel_moe_forward,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.config import get_config
from vllm.utils import FlexibleArgumentParser

import numpy as np
import matplotlib.pyplot as plt

DEFAULT_BATCH_SIZES = [
    1,
    2,
    4,
    8,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    256,
    512,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144, 
    8192, 
    10240, 
    12288, 
    14336, 
    16384, 
    20480, 
    24576, 
    28672,
    32768
]
DEFAULT_TP_SIZES = [1]

PER_ACT_TOKEN_OPTS = [False]
PER_OUT_CH_OPTS = [False]


def bench_run(
        results1: list[benchmark.Measurement],
        results2: list[benchmark.Measurement],
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
    ):

    assert dtype == torch.bfloat16, "The experimental kernel only support bfloat16"

    label = "Matmul OGS"
    sub_label = "num_experts={}, topk={}, MKN=({})".format(
        num_experts, topk, (num_tokens, hidden_size, shard_intermediate_size)
    )

    print(f"Testing: {sub_label}")

    num_expts_tot = num_experts

    randbits = [torch.randperm(num_expts_tot) for _ in range(num_tokens)]
    x = [(-1)**i * ((16384 + ((i * 512) % 4096) + bits).to(torch.int16).view(dtype)) for i, bits in enumerate(randbits)]
    exp_data = torch.stack(x).to(device="cuda")

    x = torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")
    w1 = torch.randn((num_expts_tot, shard_intermediate_size, hidden_size), dtype=dtype, device="cuda")
    w2 = torch.randn((num_expts_tot, hidden_size, shard_intermediate_size // 2), dtype=dtype, device="cuda")
    
    exp_data_tri = exp_data.clone()
    x_tri = x.clone()
    w1_tri = w1.clone()
    w2_tri = w2.clone()
    w1_tri = w1_tri.transpose(-2, -1).contiguous()
    w2_tri = w2_tri.transpose(-2, -1).contiguous()

    def run_fused_moe(
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        topk: int,
        num_repeats: int,
    ):
        for _ in range(num_repeats):
            fused_moe(
                x,
                w1,
                w2,
                router_logits,
                topk,
                renormalize=True,
            )

    def run_fused_moe_from_graph(
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        topk: int,
    ):
        with set_current_vllm_config(
            VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
        ):
            fused_moe(
                x,
                w1,
                w2,
                router_logits,
                topk,
                renormalize=True,
            )

    def run_triton_kernel(
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        topk: int,
        num_repeats: int,
    ):
        for _ in range(num_repeats):
            triton_kernel_moe_forward(
                x, w1, w2, router_logits, topk, renormalize=True
            )

    def run_triton_kernel_from_graph(
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        topk: int,
    ):
        with set_current_vllm_config(
            VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
        ):
            triton_kernel_moe_forward(
                x, w1, w2, router_logits, topk, renormalize=True
            )

    def replay_graph(graph, num_repeats):
        for _ in range(num_repeats):
            graph.replay()
        torch.cuda.synchronize()

    fused_moe_stream = torch.cuda.Stream()
    fused_moe_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(fused_moe_graph, stream=fused_moe_stream):
        run_fused_moe_from_graph(
            x,
            w1,
            w2,
            exp_data,
            topk,
        )
    torch.cuda.synchronize()

    triton_kernel_stream = torch.cuda.Stream()
    triton_kernel_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(triton_kernel_graph, stream=triton_kernel_stream):
        run_triton_kernel_from_graph(
            x_tri,
            w1_tri,
            w2_tri,
            exp_data,
            topk,
        )
    torch.cuda.synchronize()

    min_run_time = 5
    num_warmup = 5
    num_runs = 25

    globals_fused_moe = {
        # weights:
        "w1": w1,
        "w2": w2,
        "topk": topk,
        # activation
        "x": x,
        "router_logits": exp_data,
        "num_runs": num_runs,
        # Kernels
        "run_fused_moe": run_fused_moe,
        "replay_graph": replay_graph,
        # cuda graph
        "fused_moe_graph": fused_moe_graph,
    }

    globals_triton_kernel = {
        # weights:
        "w1": w1_tri,
        "w2": w2_tri,
        "topk": topk,
        # activation
        "x": x_tri,
        "router_logits": exp_data_tri,
        "num_runs": num_runs,
        # Kernels
        "run_triton_kernel": run_triton_kernel,
        "replay_graph": replay_graph,
        # cuda graph
        "triton_kernel_graph": triton_kernel_graph,
    }

    # warmup
    # run_fused_moe(
    #     x,
    #     w1,
    #     w2,
    #     exp_data,
    #     topk,
    #     num_warmup
    # )
    # results.append(
    #     benchmark.Timer(
    #         stmt="run_fused_moe(x, w1, w2, router_logits, topk, num_runs)",
    #         globals=globals_fused_moe,
    #         label=label,
    #         sub_label=sub_label,
    #         description="fused_moe",
    #     ).blocked_autorange(min_run_time=min_run_time)
    # )

    replay_graph(fused_moe_graph, num_warmup)

    results1.append(
        benchmark.Timer(
            stmt="replay_graph(fused_moe_graph, num_runs)",
            globals=globals_fused_moe,
            label=label,
            sub_label=sub_label,
            description="fused_moe_cuda_graphs",
        ).blocked_autorange(min_run_time=min_run_time)
    )

    # warmup
    # run_triton_kernel(
    #     x_tri,
    #     w1_tri,
    #     w2_tri,
    #     exp_data,
    #     topk,
    #     num_warmup
    # )

    # results.append(
    #     benchmark.Timer(
    #         stmt="run_triton_kernel(x, w1, w2, router_logits, topk, num_runs)",
    #         globals=globals_triton_kernel,
    #         label=label,
    #         sub_label=sub_label,
    #         description="triton_kernel",
    #     ).blocked_autorange(min_run_time=min_run_time)
    # )

    replay_graph(triton_kernel_graph, num_warmup)

    results2.append(
        benchmark.Timer(
            stmt="replay_graph(triton_kernel_graph, num_runs)",
            globals=globals_triton_kernel,
            label=label,
            sub_label=sub_label,
            description="triton_kernel_cuda_graphs",
        ).blocked_autorange(min_run_time=min_run_time)
    )


def get_weight_block_size_safety(config, default_value=None):
    quantization_config = getattr(config, "quantization_config", {})
    if isinstance(quantization_config, dict):
        return quantization_config.get("weight_block_size", default_value)
    return default_value


def main(args: argparse.Namespace):
    print(args)
    results_fused_moe: list[benchmark.Measurement] = []
    results_triton_kernel: list[benchmark.Measurement] = []

    config = get_config(model=args.model, trust_remote_code=args.trust_remote_code)
    config = SimpleNamespace(**config.to_dict())
    model_name = config.architectures[0] 

    if config.architectures[0] == "DbrxForCausalLM":
        E = config.ffn_config.moe_num_experts
        topk = config.ffn_config.moe_top_k
        intermediate_size = config.ffn_config.ffn_hidden_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] == "JambaForCausalLM":
        E = config.num_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] in ("DeepseekV3ForCausalLM", "DeepseekV2ForCausalLM"):
        E = config.n_routed_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] in ("Qwen2MoeForCausalLM", "Qwen3MoeForCausalLM"):
        E = config.num_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] == 'Llama4ForConditionalGeneration':
        # Support for llama4
        config =  SimpleNamespace(**config.text_config)
        E = config.num_local_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size        
    else:
        # Support for llama4
        config = config.get_text_config()
        # Default: Mixtral.
        E = config.num_local_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size

    hidden_size = config.hidden_size
    dtype = (
        torch.float16
        if current_platform.is_rocm()
        else getattr(torch, config.torch_dtype)
    )

    use_fp8_w8a8 = args.dtype == "fp8_w8a8"
    use_int8_w8a16 = args.dtype == "int8_w8a16"
    block_quant_shape = get_weight_block_size_safety(config)

    for size_m in DEFAULT_BATCH_SIZES:
        bench_run(
            results_fused_moe,
            results_triton_kernel,
            size_m,
            E,
            shard_intermediate_size,
            hidden_size,
            topk,
            dtype
        )
    
    compare = benchmark.Compare(results_triton_kernel)
    compare.print()

    labels = [str(i) for i in DEFAULT_BATCH_SIZES]
    latencies_fused_moe = [m.median * 1000 for m in results_fused_moe]
    latencies_triton_kernel = [m.median * 1000 for m in results_triton_kernel] # unit in us

    def draw_plot(labels, result1, result2, model_name, tp):
        x = np.arange(len(labels))  # label positions
        width = 0.35

        fig, ax = plt.subplots(figsize=(30, 9))
        bars1 = ax.bar(x - width/2, result1, width, label='fused_moe', color='skyblue')
        bars2 = ax.bar(x + width/2, result2, width, label='triton_kernel', color='salmon')

        # Add numbers on top of bars1
        for bar in bars1:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10
            )

        # Add numbers on top of bars2
        for bar in bars2:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10
            )

        # ax.set_ylim(0, ylim)
        ax.set_xlabel("Batch size", fontsize=20)
        ax.set_ylabel("Latency (ms)", fontsize=20)
        ax.set_title(f"{model_name} with tp={tp}", fontsize=20)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=15)
        ax.legend(fontsize=25)
        # ax.set_yticklabels(ontsize=15)
        plt.tick_params(axis='y', labelsize=20)
        plt.tight_layout()
        plt.savefig(f"{model_name}-tp={tp}.pdf")

    draw_plot(labels, latencies_fused_moe, latencies_triton_kernel, model_name, args.tp_size)

if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark new Triton Kernel across specified models/shapes/batches"
    )
    parser.add_argument(
        "--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    parser.add_argument(
        "--tp-size", "-tp", "--tensor-parallel-size", type=int, default=1
    )
    parser.add_argument(
        "--dtype", type=str, choices=["auto", "fp8_w8a8", "int8_w8a16"], default="auto"
    )
    parser.add_argument("--trust-remote-code", action="store_true")

    args = parser.parse_args()
    main(args)
