# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from vllm.model_executor.models.mlp_speculator import MLPSpeculator
from vllm.v1.sample.metadata import SamplingMetadata


class MLPProposer:

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.num_speculative_tokens = vllm_config.speculative_config.num_speculative_tokens

    def load_model(self, target_model: nn.Module):
        loader = get_model_loader(self.vllm_config.load_config)
        draft_model_config = \
            self.vllm_config.speculative_config.draft_model_config
        # FIXME: This does not handle with distributed inference.
        target_device = self.vllm_config.device_config.device
        with set_default_torch_dtype(
                draft_model_config.dtype), set_current_vllm_config(
                    self.vllm_config):
            self.model = MLPSpeculator(
                vllm_config=self.vllm_config).to(target_device)

        loaded_weights = self.model.load_weights(
            loader.get_all_weights(
                self.vllm_config.speculative_config.draft_model_config,
                self.model))

    def propose(self, input_ids: torch.Tensor,
                previous_hidden_states: torch.Tensor,
                sampling_metadata: SamplingMetadata):
        sample_output = self.model.generate_proposals(
            input_ids, previous_hidden_states, self.num_speculative_tokens,
            sampling_metadata)
        sampled_token_ids = torch.stack(
            [
                sampler_output.sampled_token_ids.flatten()
                for sampler_output in sample_output
            ],
            dim=0,
        )
        sampled_token_ids = sampled_token_ids.transpose(0, 1)

        return sampled_token_ids
