script_path="$(realpath "$0")"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc-per-node 8  torchrun_main.py \
    --model_config configs/llama_1b.json \
    --batch_size 8 \
    --total_batch_size 320 \
    --lr 5e-4 \
    --max_length 1024 \
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
    --continue_from /data/harsha/relora_exps_olvi2/model1b/warmup_20k_gaccum5/model_5000 \
    --tags relora_1b \
    --exp_name tff_1b_40k_8gpus_lr5em4_gaccum5_int5460 \
    --script_path $script_path

