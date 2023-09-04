# script_path="$(realpath "$0")"
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc-per-node 4 torchrun_main.py \
#    --model_config configs/llama_1b.json \
#    --batch_size 8 \
#    --total_batch_size 320 \
#    --lr 5e-4 \
#    --max_length 1024 \
#    --tags warm_start_1B \
#    --save_every 3000 \
#    --num_training_steps 40000 \
#    --exp_name warmup_1b_40k_4gpus_lr5em4_gaccum5_int5460_b8
#    --script_path $script_path

# first 40k
script_path="$(realpath "$0")"
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc-per-node 4  torchrun_main.py \
    --model_config configs/llama_1b.json \
    --batch_size 8 \
    --total_batch_size 320 \
    --lr 5e-4 \
    --max_length 1024 \
    --use_peft \
    --retff 2000 \
    --cycle_length 2000 \
    --restart_warmup_steps 100 \
    --scheduler cosine_restarts \
    --warmup_steps 500 \
    --reset_optimizer_on_retff True \
    --num_training_steps 40000 \
    --save_every 3000 \
    --eval_every 3000 \
    --continue_from /data/harsha/relora_exps_olvi2/model1b/warmup_40k_gaccum10/model_6000 \
    --tags relora_1b \
    --exp_name tff_1b_40k_4gpus_lr5em4_gaccum10_int5460_retff2k_scal32b512 \
    --script_path $script_path \
    --scaling 0.0625 \
    --k_attn 4 \
    --l_attn 512 \
    --n_attn 2048 \
    --k_mlp 12 \
    --l_mlp 512 \
    --n_mlp 5460 \

