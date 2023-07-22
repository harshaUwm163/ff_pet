import os
import math
import json
from typing import List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, LlamaForCausalLM, AutoConfig
from .construct_tff import construct_real_tff


@dataclass
class ReTffConfig:
    tff_dropout: float
    target_modules: List[str]
    keep_original_weights: bool
    tff_only: bool = False
    trainable_scaling: bool = False


class ReTffModel(torch.nn.Module):
    def __init__(self, model, tff_dropout, target_modules, keep_original_weights=True, tff_only=False, trainable_scaling=False):

        super().__init__()
        self.wrapped_model: nn.Module = model
        self.tff_dropout = tff_dropout
        self.target_modules = target_modules
        self.keep_original_weights = keep_original_weights
        self.tff_only = tff_only
        self.trainable_scaling = trainable_scaling

        self._config = ReTffConfig(
            tff_dropout=tff_dropout,
            target_modules=target_modules,
            keep_original_weights=keep_original_weights,
        )

        self.tffs_dict = {}
        # generate the TFFs
        self.k_attn = 3
        self.l_attn = 256
        self.n_attn = 768
        self.tffs_dict['all_for_one'] = construct_real_tff(self.k_attn, self.l_attn // 2, self.n_attn // 2).permute(0,2,1)

        self.k_mlp = 16
        self.l_mlp = 128
        self.n_mlp = 2048
        self.tffs_dict['mlp'] = construct_real_tff(self.k_mlp, self.l_mlp // 2, self.n_mlp // 2).permute(0,2,1)

        # self.k_attn = 16
        # self.l_attn = 8
        # self.n_attn = 128
        # self.tffs_dict['all_for_one'] = construct_real_tff(self.k_attn, self.l_attn // 2, self.n_attn // 2).permute(0,2,1)

        # self.k_mlp = 16
        # self.l_mlp = 22
        # self.n_mlp = 352
        # self.tffs_dict['mlp'] = construct_real_tff(self.k_mlp, self.l_mlp // 2, self.n_mlp // 2).permute(0,2,1)

        # patch methods
        self.forward = self.wrapped_model.forward

        target_modules_list = target_modules
        if isinstance(target_modules_list, str):
            target_modules_list = [target_modules_list]

        for module_name, module in self.wrapped_model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue

            target_key = 'all_for_one'
            if 'mlp' in module_name and 'down' not in module_name:
                target_key = 'mlp'
            
            new_module = ReTffLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                frame = self.tffs_dict[target_key][0], # init with the first frame
                tff_dropout=self.tff_dropout,
                tff_only=self.tff_only,
            )
            if self.keep_original_weights:
                new_module.weight = module.weight
                if module.bias is not None:
                    new_module.bias = module.bias

            if self.tff_only:
                assert not self.keep_original_weights
                module.weight = None

            parent = self._get_parent(module_name)
            module_suffix = module_name.split(".")[-1]
            setattr(parent, module_suffix, new_module)

    def _get_parent(self, module_name):
        module_names_list = module_name.split(".")
        parent_name = ".".join(module_names_list[:-1])
        parent = self.wrapped_model.get_submodule(parent_name)
        return parent

    def merge_and_reinit(self, device):
        for module_name, module in self.named_modules():
            if isinstance(module, ReTffLinear):
                if 'mlp' in module_name and 'down' not in module_name:
                    target_key = 'mlp'
                    new_tff_index = torch.randint(self.k_mlp,(1, ))[0].item()
                else:
                    target_key = 'all_for_one'
                    new_tff_index = torch.randint(self.k_attn,(1, ))[0].item()

                module.merge_and_reinit(new_frame=self.tffs_dict[target_key][new_tff_index], device=device)

    def save_pretrained(self, path):
        self.wrapped_model.save_pretrained(path)
        with open(os.path.join(path, "reTff_config.json"), "w") as f:
            json.dump(self._config.__dict__, f, indent=4)

    @classmethod
    def from_pretrained(cls, path):
        with open(os.path.join(path, "reTff_config.json"), "r") as f:
            reTff_config = json.load(f)

        config = AutoConfig.from_pretrained(path)

        base_model = AutoModelForCausalLM.from_config(config)
        if "keep_original" in reTff_config:
            print("WARNING: keep_original is deprecated. Use tff_only instead.")
            print(f"keep_original: {reTff_config['keep_original']}")
            reTff_config["tff_only"] = not reTff_config.pop("keep_original")
            reTff_config["keep_original_weights"] = not reTff_config["tff_only"]

        if "trainable_scaling" not in reTff_config:
            reTff_config["trainable_scaling"] = False

        model = cls(base_model, **reTff_config)

        with open(os.path.join(path, "pytorch_model.bin"), "rb") as f:
            state_dict = torch.load(f, map_location="cpu")

        model.wrapped_model.load_state_dict(state_dict, strict=True)
        return model


# The code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
class ReTffLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        frame: torch.FloatTensor, 
        tff_dropout: float = 0.1,
        tff_only: bool = False,
        **kwargs,
    ):
        """Wraps linear layer x W into x W + x W_a @ W_b * lora_alpha / r
        
        Notice that scale = lora_alpha / r.
        """

        if not tff_only:
            # if full model weight + tff weight
            nn.Linear.__init__(self, in_features, out_features, **kwargs)
        else:
            nn.Module.__init__(self)
            self.weight = None
            self.bias = None

        self.in_features = in_features
        self.out_features = out_features
        self.tff_dropout = nn.Dropout(p=tff_dropout)
        self.tff_only = tff_only

        # get the dimension of the subspace
        self.l = frame.shape[-1]
        assert frame.shape[0] == self.out_features, "the frame dimension and output dimension must match"

        # the A matrix is going to be learned
        self.tff_A = nn.Linear(in_features, self.l, bias=False)
        # the B matrix is fixed to the frame matrix
        self.proj_B = nn.Linear(self.l, out_features, bias=False)
        self.proj_B.weight = nn.Parameter(frame, requires_grad = False)

        # Freezing the pre-trained weight matrix
        if not self.tff_only:
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        if not hasattr(self, "tff_A"):
            # we are in nn.Linear calling reset_parameters
            nn.Linear.reset_parameters(self)
            return

        if not self.tff_only:
            nn.init.zeros_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

        # disregard the B as its the frames. A should be init to random values
        nn.init.kaiming_uniform_(self.tff_A.weight, a=math.sqrt(5))
    
    @torch.no_grad()
    def merge_and_reinit(self, new_frame = None, device = torch.device('cpu')):
        if self.tff_only:
            print("WARNING: Skipping merge and reinit, because only TFF parameters are used")
            return

        self.weight.data += self.proj_B.weight @ self.tff_A.weight
        self.merged = False
        nn.init.zeros_(self.tff_A.weight)
        # nn.init.kaiming_uniform_(self.tff_A.weight, a=math.sqrt(5))
        # update the frame as well
        if new_frame is not None:
            self.proj_B.weight = torch.nn.Parameter(new_frame.type(self.tff_A.weight.type()).to(device), requires_grad = False)

    def forward(self, x: torch.Tensor):
        if self.tff_only:
            # just Tff
            return self.proj_B(self.tff_A(self.tff_dropout(x)))

        result = F.linear(x, self.weight, bias=self.bias)

        result += self.proj_B(self.tff_A(self.tff_dropout(x)))
        return result
