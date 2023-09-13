script_path="$(realpath "$0")"
echo "running the python torch run"
CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.run --nproc-per-node 1 torchrun_main.py \
    --model_config configs/llama_130m.json \
    --batch_size 32 \
    --total_batch_size 32 \
    --lr 1e-3 \
    --max_length 512 \
    --use_peft \
    --retff 10 \
    --cycle_length 10 \
    --restart_warmup_steps 1 \
    --scheduler cosine_restarts \
    --warmup_steps 1 \
    --reset_optimizer_on_retff True \
    --num_training_steps 40000 \
    --save_every 150 \
    --eval_every 150 \
    --tags relora_130M \
    --exp_name debug_thread \
    --script_path $script_path \
    --scaling 1.0 \
    --num_frames 8 \
    --k_attn 24 \
    --l_attn 32 \
    --n_attn 768 \
    --k_mlp 128 \
    --l_mlp 16 \
    --n_mlp 2048 \

echo "finished running the python torch "

    # for num params close to relora
    # --k_attn 3 \
    # --l_attn 256 \
    # --n_attn 768 \
    # --k_mlp 16 \
    # --l_mlp 128 \
    # --n_mlp 2048 \

    # for mf test
    # --k_attn 24 \
    # --l_attn 32 \
    # --n_attn 768 \
    # --k_mlp 128 \
    # --l_mlp 16 \
    # --n_mlp 2048 \