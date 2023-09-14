import os
import time
import json
import random
import hashlib
import argparse
from typing import Union
from functools import partial

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM

import datasets
import datasets.distributed
import wandb

from tqdm import tqdm
from loguru import logger

from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset
from peft_pretraining.modeling_llama import LlamaForCausalLM
# from peft_pretraining.relora import ReLoRaModel, ReLoRaLinear
from peft_pretraining.re_tff import ReTffModel, ReTffLinear
from peft_pretraining.construct_tff import construct_real_tff

transformers.logging.set_verbosity_error()

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--continue_from_peft", type=str, default=None, help="Continue training with ReTff, loading optimizer and scheduler from the checkpoint.")
    parser.add_argument("--restore_optimizer", default=False, action="store_true")

    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)

    parser.add_argument("--use_peft", action="store_true")
    # parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--retff", type=int, default=None)
    parser.add_argument("--train_scaling", default=False, action="store_true")
    parser.add_argument("--reset_optimizer_on_retff", default=True, type=lambda x: x.lower() == "true")
    parser.add_argument("--optimizer_random_pruning", default=0.0, type=float,
                        help="Use random pruning to reduce optimizer matrix internal dimensionality.")
    parser.add_argument("--optimizer_magnitude_pruning", default=0.0, type=float,
                        help="Use magnitude pruning to reduce optimizer matrix internal dimensionality.")
    parser.add_argument("--force_keep_original", default=False, action="store_true",
                        help=("Keep original model parameters even if retff is None. "
                              "Useful for making sure that full-Tff model is equivalent to model+Tff."))

    parser.add_argument("--train_ln", default=True, action="store_true")
    parser.add_argument("--optimizer", default="Adam", choices=["Adam", "Shampoo"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--cycle_length", type=int, default=None, help="Number of steps per cycle for cosine scheduler")
    parser.add_argument("--restart_warmup_steps", type=int, default=None, help="Number of steps for cosine restarts (only used for cosine_restarts)")
    parser.add_argument("--adjust_step", type=int, default=0, help="Number of steps to adjust the scheduler by. "
                            f"Useful when you want to sync ReTff resets with the scheduler for a warmed up model. "
                            f"You need to use it, when your warmup_step % retff_resets != 0")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)

    parser.add_argument("--eval_every", type=int, default=5_000)

    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--num_scheduling_steps", type=int, default=None,
                        help="Number of **scheduler steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=8)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_name", type=str, default='debug_thread', help='name of the experiments')

    parser.add_argument("--script_path", type=str, default=None, help='path of the script')
    parser.add_argument("--scaling", type=float, default=1.0, help='scaling after applying re_tff')

    parser.add_argument("--k_attn", type=int, default=4)
    parser.add_argument("--l_attn", type=int, default=512)
    parser.add_argument("--n_attn", type=int, default=2048)
    parser.add_argument("--k_mlp", type=int, default=12)
    parser.add_argument("--l_mlp", type=int, default=512)
    parser.add_argument("--n_mlp", type=int, default=5460)

    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--guide_after_n_restarts", type=int, default=10)

    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)

    if args.continue_from_peft and not args.restore_optimizer:
        logger.warning("--continue_from_peft is set, but --restore_optimizer is not. "
                       "This means that you will train with the optimizer from the checkpoint, "
                       "but will not save the optimizer state. "
                       "This is probably not what you want.")

    return args

def main(args):
    # seed all
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_rank)
    print(f"local rank: {args.local_rank}, device: {torch.cuda.current_device()}")

    # assumes that we are using a single node
    torch.distributed.init_process_group(
        backend="nccl",
        rank=args.local_rank,
        world_size=torch.cuda.device_count()
    )

    # flag for moving the file
    moved_script = False

    global_rank = torch.distributed.get_rank()
    local_rank = global_rank % torch.cuda.device_count()
    world_size = torch.distributed.get_world_size()
    device = f"cuda:{local_rank}"

    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"
    else:
        args.gradient_accumulation = 1
        args.total_batch_size = world_size * args.batch_size

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # turn off logger
    if global_rank != 0: logger.remove()

    # initialize wandb without config (it is passed later)
    if global_rank == 0:
        wandb.init( project="peft_pretraining", 
                    tags=args.tags, 
                    mode = 'offline' if args.exp_name == 'debug_thread' else 'online', 
                    name = args.exp_name,
                    )

    logger.info(f"Using torch.distributed with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    dataset_name = "c4"
    assert dataset_name == "c4"
    data = datasets.load_dataset("c4", "en", split="train", streaming=True)

    # this seed is hard-coded to guarantee the same order of the examples (for any --seed)
    seed_for_shuffle = 42
    if args.continue_from is not None:
        # add hash of the path to the checkpoint to the seed
        seed_for_shuffle += int(hashlib.sha256(args.continue_from.encode("utf-8")).hexdigest(), 16) % 10**8
    if args.continue_from_peft is not None:
        seed_for_shuffle += int(hashlib.sha256(args.continue_from_peft.encode("utf-8")).hexdigest(), 16) % 10**8

    logger.info(f"Shuffling data with seed {seed_for_shuffle} (should be 42 for the first run and 42 + hash(checkpoint_path) for the runs that continue from a checkpoint)")
    data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)
    data = datasets.distributed.split_dataset_by_node(
        data, rank=global_rank, world_size=world_size,
    )

    # it doesn't matter which tokenizer we use, because we train from scratch
    # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    dataset = PreprocessedIterableDataset(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)

    model_config = AutoConfig.from_pretrained(args.model_config)
    if args.use_hf_model:
        model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
    else:
        model = LlamaForCausalLM(model_config)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    global_step = 0
    update_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")
        checkpoint_path = os.path.join(args.continue_from, "pytorch_model.bin")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
        logger.info(f"Model successfully loaded (strict=True policy)")

        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}")
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(f"Will train for {args.num_training_steps - update_step} update steps")
        else:
            logger.warning(f"Did not find training state in {args.continue_from}, global step will start from zero")
        logger.info("*" * 40)

    import matplotlib.pyplot as plt
    import math

    kvals = [3,6,12,24,48,96,192,384]
    weight_mat = model.model.layers[5].self_attn.k_proj.weight.data
    num_frames_keep = 0.7 # 70%
    n = 768
    for k in kvals:
        l = n // k
        # set the plots 
        fig, axes = plt.subplots(nrows=1, ncols=2)

        # plot the energies in the TFF subspaces
        tffs = construct_real_tff(k, l//2, n//2)
        projs = torch.matmul(tffs, weight_mat)
        norms = torch.norm(projs, p='fro', dim=(1,2))
        ax1 = axes[0]
        ax1.stem(norms.cpu().numpy())
        ax1.set_title(f'tffs_k_{k}_l_{l}')

        # plot the energies in the Canonical subspaces
        eig_frames = torch.eye(n).view(k, l, n)
        projs = torch.matmul(eig_frames, weight_mat)
        norms = torch.norm(projs, p='fro', dim=(1,2))
        ax2 = axes[1]
        ax2.stem(norms.cpu().numpy())
        ax2.set_title(f'canon_k_{k}_l_{l}')
        plt.savefig(os.path.join('./plots', f'k_{k}_l_{l}.png'))

        ### approximation capabilities
        kmax = math.floor(k * num_frames_keep)
        used_frames = tffs[:kmax,...]
        approx_proj_mat = torch.matmul(used_frames.permute(0,2,1), used_frames).sum(dim=0)
        low_rank_approx = approx_proj_mat @ weight_mat
        approx_err = weight_mat - low_rank_approx
        num_params = n**2
        wt_norm = torch.norm(weight_mat)
        sparsities = []
        sparse_errs = []
        for thresh in np.linspace(approx_err.abs().min(), approx_err.abs().max(), 10):
            mask = approx_err.abs().ge(thresh)
            sparse_approx = approx_err * mask.float()
            sparsity_percent = 100 - mask.sum()/num_params * 100
            sparse_approx_err = torch.norm(approx_err - sparse_approx)/wt_norm * 100

            sparsities.append(sparsity_percent)
            sparse_errs.append(sparse_approx_err)
        
        plt.figure()
        plt.plot(sparsities, sparse_errs)
        plt.title(f'err vs sparse approx_k_{k}_l_{l}')
        plt.savefig(os.path.join('./plots', 'sparse_approx', f'sparse_err_k_{k}_l_{l}.png'))


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)
