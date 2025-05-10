# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.model_loader.loader import get_model_loader
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from vllm.model_executor.models.medusa import Medusa
from vllm.v1.sample.metadata import SamplingMetadata


class MedusaProposer:

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.num_speculative_tokens = (
            vllm_config.speculative_config.num_speculative_tokens)

    def propose(self, previous_hidden_states,
                sampling_metadata: SamplingMetadata):
        # from vllm.model_executor.sampling_metadata import SamplingMetadata
        logits = self.model.compute_logits(
            hidden_states=self.model.forward(previous_hidden_states),
            sampling_metadata=None)

        logits = torch.stack(logits, dim=0).float()
        token_ids = logits.argmax(-1)
        token_ids = token_ids.transpose(0, 1)

        return token_ids

    def load_model(self, target_model: nn.Module):

        loader = get_model_loader(self.vllm_config.load_config)
        draft_model_config = \
            self.vllm_config.speculative_config.draft_model_config
        # FIXME: This does not handle with distributed inference.
        target_device = self.vllm_config.device_config.device
        with set_default_torch_dtype(
                draft_model_config.dtype), set_current_vllm_config(
                    self.vllm_config):
            self.model = Medusa(vllm_config=self.vllm_config).to(target_device)

        loaded_weights = self.model.load_weights(
            loader.get_all_weights(
                self.vllm_config.speculative_config.draft_model_config,
                self.model))
