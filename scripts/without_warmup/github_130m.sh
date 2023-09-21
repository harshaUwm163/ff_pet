# CUDA_VISIBLE_DEVICES=1,2,3,4,5 python -m torch.distributed.run --nproc-per-node 5 torchrun_main.py \
#    --model_config configs/llama_130m.json \
#    --batch_size 75 \
#    --total_batch_size 750 \
#    --lr 1e-3 \
#    --max_length 256 \
#    --tags warm_start_130M \
#    --save_every 1000 \
#    --num_training_steps 40000 \
#    --exp_name warmup4tff_130m_40k_5gpus_lr1em3_gaccum2

# 
script_path="$(realpath "$0")"
CUDA_VISIBLE_DEVICES=1,5 python -m torch.distributed.run --nproc-per-node 2  torchrun_main.py \
    --model_config configs/llama_130m.json \
    --batch_size 128 \
    --total_batch_size 256 \
    --lr 1e-3 \
    --max_length 256 \
    --use_peft \
    --retff 1000 \
    --cycle_length 1000 \
    --restart_warmup_steps 100 \
    --scheduler cosine_restarts \
    --warmup_steps 500 \
    --reset_optimizer_on_retff True \
    --num_training_steps 40000 \
    --save_every 1000 \
    --eval_every 1000 \
    --tags relora_130M \
    --exp_name rr_tff_130m_k2_itr2 \
    --script_path $script_path \
    --guide_after_n_restarts 20000000 \
    --scaling 1.0 \
    --num_frames 1 \
    --k_attn 2 \
    --l_attn 384 \
    --n_attn 768 \
    --k_mlp 2 \
    --l_mlp 1024 \
    --n_mlp 2048 \
    --merged_model_path /data/harsha/ff_pet/checkpoints/llama_130m-2023-09-18-04-28-35/ReTffModel_1.pt \

    # for num params close to relora
    # --num_frames 1 \
    # --k_attn 3 \
    # --l_attn 256 \
    # --n_attn 768 \
    # --k_mlp 16 \
    # --l_mlp 128 \
    # --n_mlp 2048 \

    # works pretty well
    # --num_frames 10 \
    # --k_attn 24 \
    # --l_attn 32 \
    # --n_attn 768 \
    # --k_mlp 128 \
    # --l_mlp 16 \
    # --n_mlp 2048 \

    # best so far
    # --num_frames 15 \
    # --k_attn 48 \
    # --l_attn 16 \
    # --n_attn 768 \
    # --k_mlp 64 \
    # --l_mlp 32 \
    # --n_mlp 2048 \

    # works, but previous is better
    # --num_frames 14 \
    # --k_attn 96 \
    # --l_attn 8 \
    # --n_attn 768 \
    # --k_mlp 32 \
    # --l_mlp 64 \
    # --n_mlp 2048 \
