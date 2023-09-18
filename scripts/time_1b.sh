# script_path="$(realpath "$0")"
# CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.run --nproc-per-node 2 torchrun_main.py \
#    --model_config configs/llama_1b.json \
#    --batch_size 8 \
#    --total_batch_size 16 \
#    --lr 5e-4 \
#    --max_length 1024 \
#    --tags warm_start_1B \
#    --save_every 5000 \
#    --num_training_steps 40000 \
#    --exp_name time_1b_fullRank \
#    --script_path $script_path

# first 40k
script_path="$(realpath "$0")"
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.run --nproc-per-node 2  torchrun_main.py \
    --model_config configs/llama_1b.json \
    --batch_size 8 \
    --total_batch_size 16 \
    --lr 5e-4 \
    --max_length 1024 \
    --use_peft \
    --retff 50000 \
    --cycle_length 50000 \
    --restart_warmup_steps 100 \
    --scheduler cosine_restarts \
    --warmup_steps 500 \
    --reset_optimizer_on_retff True \
    --num_training_steps 400000 \
    --save_every 50000 \
    --eval_every 50000 \
    --tags relora_1b \
    --exp_name time_1b_tff_attn_mlp \
    --script_path $script_path \
    --scaling 1.0 \
    --num_frames 15 \
    --k_attn 128 \
    --l_attn 16 \
    --n_attn 2048 \
    --k_mlp 172 \
    --l_mlp 32 \
    --n_mlp 5460 \

#     # --scaling 0.0625 \
#     # --k_attn 4 \
#     # --l_attn 512 \
#     # --n_attn 2048 \
#     # --k_mlp 12 \
#     # --l_mlp 512 \
#     # --n_mlp 5460 \

