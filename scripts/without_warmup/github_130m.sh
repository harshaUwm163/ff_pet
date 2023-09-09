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
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc-per-node 8  torchrun_main.py \
    --model_config configs/llama_130m.json \
    --batch_size 32 \
    --total_batch_size 256 \
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
    --save_every 1000 \
    --eval_every 1000 \
    --tags relora_130M \
    --exp_name tff_130m_40k_8gpus_lr1em3_gaccum1_retff1k_scal1p0 \
    --script_path $script_path \
    --scaling 1.0 \
    --k_attn 3 \
    --l_attn 256 \
    --n_attn 768 \
    --k_mlp 16 \
    --l_mlp 128 \
    --n_mlp 2048 \
