# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

from vllm import LLM, SamplingParams

os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
os.environ["VLLM_USE_FLASHINFER_MOE_FP8"] = "0"
os.environ["VLLM_DEEP_GEMM_WARMUP"] = "skip"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.expanduser("~/.cache/torchinductor")

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

prompt_token_ids = [
    {
        "prompt_token_ids": [
            0,
            128803,
            671,
            6102,
            294,
            8760,
            344,
            11111,
            14,
            305,
            270,
            6102,
            294,
            10322,
            344,
            128804,
        ],
    },  # The capital of France is Paris, and the capital of Germany is
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, max_tokens=1, logprobs=5)


MODEL_CKPT = "/mnt/lustre/deepseek_v4-2026-04/ckpt20260409/DeepSeek-V4-HF-FP4/"


def main():
    # Create an LLM.
    llm = LLM(
        model=MODEL_CKPT,
        # enforce_eager=True,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.9,
        block_size=256,
        kv_cache_dtype="fp8",
        # To use this: uv pip install fastsafetensors
        # Weight loading takes 40-50% less time
        # load_format="fastsafetensors",
        compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"},
        enforce_eager=True,
        kernel_config={"enable_flashinfer_autotune": False},
    )
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompt_token_ids, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        print(f"Prompt:    {prompt!r}")
        generated_text = output.outputs[0].text
        print(f"Output:    {generated_text!r}")
        for logit, logprob in output.outputs[0].logprobs[0].items():
            print(f"Logit No: {logit}")
            print(f"Logprob: {logprob.logprob}")
        print("-" * 60)


def test():
    main()


if __name__ == "__main__":
    main()
