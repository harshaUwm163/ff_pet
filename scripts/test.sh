script_path="$(realpath "$0")"
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.run --nproc-per-node 1  torchrun_main.py \
    --model_config configs/llama_130m.json \
    --batch_size 64 \
    --total_batch_size 64 \
    --lr 1e-3 \
    --max_length 512 \
    --use_peft \
    --retff 1000 \
    --cycle_length 1000 \
    --restart_warmup_steps 100 \
    --scheduler cosine_restarts \
    --warmup_steps 500 \
    --reset_optimizer_on_retff True \
    --num_training_steps 40000 \
    --save_every 3000 \
    --eval_every 3000 \
    --tags relora_130M \
    --exp_name tff_130m_40k_4gpus_lr1em3_gaccum1_retff1k_scal1p0 \
    --script_path $script_path \
    --scaling 1.0 \
    --k_attn 3 \
    --l_attn 256 \
    --n_attn 768 \
    --k_mlp 16 \
    --l_mlp 128 \
    --n_mlp 2048 \