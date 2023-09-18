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
    def __init__(self, model, tff_dropout, target_modules, keep_original_weights=True, tff_only=False, trainable_scaling=False, scaling = 1.0, 
        k_attn = 4,
        l_attn = 512,
        n_attn = 2048,
        k_mlp = 12,
        l_mlp = 512,
        n_mlp = 5460,
        num_frames = 1,
        ):

        super().__init__()
        self.wrapped_model: nn.Module = model
        self.tff_dropout = tff_dropout
        self.target_modules = target_modules
        self.keep_original_weights = keep_original_weights
        self.tff_only = tff_only
        self.trainable_scaling = trainable_scaling
        self.scaling = scaling
        # self.num_frames = num_frames

        self._config = ReTffConfig(
            tff_dropout=tff_dropout,
            target_modules=target_modules,
            keep_original_weights=keep_original_weights,
        )

        self.tffs_dict = {}
        # # generate the TFFs 130m params
        # self.k_attn = 3
        # self.l_attn = 256
        # self.n_attn = 768
        # self.tffs_dict['all_for_one'] = construct_real_tff(self.k_attn, self.l_attn // 2, self.n_attn // 2).permute(0,2,1)

        # self.k_mlp = 16
        # self.l_mlp = 128
        # self.n_mlp = 2048
        # self.tffs_dict['mlp'] = construct_real_tff(self.k_mlp, self.l_mlp // 2, self.n_mlp // 2).permute(0,2,1)

        # # generate the TFFs 250m params
        # self.k_attn = 3
        # self.l_attn = 256
        # self.n_attn = 768
        # self.tffs_dict['all_for_one'] = construct_real_tff(self.k_attn, self.l_attn // 2, self.n_attn // 2).permute(0,2,1)

        # self.k_mlp = 20
        # self.l_mlp = 128
        # self.n_mlp = 2560
        # self.tffs_dict['mlp'] = construct_real_tff(self.k_mlp, self.l_mlp // 2, self.n_mlp // 2).permute(0,2,1)

        # # generate the TFFs 350m params
        # self.k_attn = 4
        # self.l_attn = 512
        # self.n_attn = 1024
        # self.tffs_dict['all_for_one'] = construct_real_tff(self.k_attn, self.l_attn // 2, self.n_attn // 2).permute(0,2,1)

        # self.k_mlp = 12
        # self.l_mlp = 256
        # self.n_mlp = 2736
        # self.tffs_dict['mlp'] = construct_real_tff(self.k_mlp, self.l_mlp // 2, self.n_mlp // 2).permute(0,2,1)

        # # generate the TFFs 1b params
        # self.k_attn = 4
        # self.l_attn = 512
        # self.n_attn = 2048
        # self.tffs_dict['all_for_one'] = construct_real_tff(self.k_attn, self.l_attn // 2, self.n_attn // 2).permute(0,2,1)

        # self.k_mlp = 12
        # self.l_mlp = 512
        # self.n_mlp = 5460
        # self.tffs_dict['mlp'] = construct_real_tff(self.k_mlp, self.l_mlp // 2, self.n_mlp // 2).permute(0,2,1)

        # generate the TFFs 1b params
        self.k_attn = k_attn
        self.l_attn = l_attn
        self.n_attn = n_attn
        self.tffs_dict['all_for_one'] = construct_real_tff(self.k_attn, self.l_attn // 2, self.n_attn // 2).permute(0,2,1)

        self.k_mlp = k_mlp
        self.l_mlp = l_mlp
        self.n_mlp = n_mlp
        self.tffs_dict['mlp'] = construct_real_tff(self.k_mlp, self.l_mlp // 2, self.n_mlp // 2).permute(0,2,1)

        # patch methods
        self.forward = self.wrapped_model.forward

        # get the current frame index
        self.curr_frame_id = 1

        target_modules_list = target_modules
        if isinstance(target_modules_list, str):
            target_modules_list = [target_modules_list]

        for module_name, module in self.wrapped_model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue

            target_key = 'all_for_one'
            k_val = self.k_attn
            if 'mlp' in module_name and 'down' not in module_name:
                target_key = 'mlp'
                k_val = self.k_mlp

            # # using multiple frames at a time
            # init_frame_indices = torch.randperm(k_val)[:self.num_frames]
            # init_frames = self.tffs_dict[target_key][init_frame_indices]
            # init_frames = torch.cat(init_frames.unbind(), dim = 1)
            # # using a single frame at a time
            # init_frames = self.tffs_dict[target_key][0]
            # # rank ramp
            init_frames = torch.cat(self.tffs_dict[target_key][:self.curr_frame_id].unbind(), dim=1)
            
            new_module = ReTffLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                frame = init_frames, # init with the first frame
                tff_dropout=self.tff_dropout,
                tff_only=self.tff_only,
                scaling = self.scaling
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

    def merge_and_reinit(self, device, draw_rand = True, num_frames_incr = 1):
        updated_indices = {}
        if self.curr_frame_id < self.k_attn:
            for module_name, module in self.named_modules():
                if isinstance(module, ReTffLinear):
                    if 'mlp' in module_name and 'down' not in module_name:
                        target_key = 'mlp'
                    else:
                        target_key = 'all_for_one'

                    new_frames = torch.cat(self.tffs_dict[target_key][self.curr_frame_id:self.curr_frame_id+num_frames_incr].unbind(), 
                                           dim=1)
                    module.merge_and_reinit(new_frame=new_frames, device=device)

            self.curr_frame_id += num_frames_incr
        return updated_indices

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
class ReTffLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        frame: torch.FloatTensor, 
        tff_dropout: float = 0.1,
        tff_only: bool = False,
        scaling: float = 1.0,
        **kwargs,
    ):
        """Wraps linear layer x W into x W + x W_a @ W_b * lora_alpha / r
        
        Notice that scale = lora_alpha / r.
        """

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.tff_dropout = nn.Dropout(p=tff_dropout)
        self.tff_only = tff_only
        self.scaling = scaling

        # get the dimension of the subspace
        self.l = frame.shape[-1]
        assert frame.shape[0] == self.out_features, "the frame dimension and output dimension must match"

        # the A matrix is going to be learned
        self.tff_A = nn.Linear(in_features, self.l, bias=False)
        # the B matrix is fixed to the frame matrix
        self.proj_B = nn.Linear(self.l, out_features, bias=False)
        self.proj_B.weight = nn.Parameter(frame, requires_grad = False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.tff_A.weight, a=math.sqrt(5))
    
    @torch.no_grad()
    def merge_and_reinit(self, new_frame = None, device = torch.device('cpu')):
        if self.tff_only:
            print("WARNING: Skipping merge and reinit, because only TFF parameters are used")
            return

        # update the frame as well
        if new_frame is not None:
            d_type = self.proj_B.weight.type()
            curr_l = new_frame.shape[-1]
            # update B by appending the frames
            self.proj_B.weight.data = torch.cat((self.proj_B.weight.data, new_frame.type(d_type)), dim = 1)
            # extend A with zeros since we are increasing the frames
            self.tff_A.weight.data = torch.cat((self.tff_A.weight.data, torch.zeros(curr_l, self.in_features).type(d_type)),
                                                dim=0)

    def forward(self, x: torch.Tensor):
        if self.tff_only:
            # just Tff
            return self.proj_B(self.tff_A(self.tff_dropout(x))) * self.scaling

        dropped_x = self.tff_dropout(x)

        result = self.proj_B(self.tff_A(dropped_x)) * self.scaling
        return result
