# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc-per-node 8 torchrun_main.py \
#    --model_config configs/llama_1b.json \
#    --batch_size 8 \
#    --total_batch_size 320 \
#    --lr 5e-4 \
#    --max_length 1024 \
#    --tags warm_start_1B \
#    --save_every 1000 \
#    --num_training_steps 20000 \
#    --exp_name warmup_1b_20k_8gpus_lr5em4_gaccum5_int5460

# first 20k
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc-per-node 8  torchrun_main.py \
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
    --num_training_steps 20000 \
    --save_every 3000 \
    --eval_every 3000 \
    --continue_from /data/harsha/relora_exps_olvi2/model1b/warmup_20k_gaccum5/model_5000 \
    --tags relora_1b \
    --exp_name tff_1b_20k_8gpus_lr5em4_gaccum5_int5460

# # Olvi 2
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc-per-node 8  torchrun_main.py \
#     --model_config configs/llama_130m.json \
#     --batch_size 75 \
#     --total_batch_size 1200 \
#     --lr 1e-3 \
#     --max_length 512 \
#     --use_peft \
#     --retff 500 \
#     --cycle_length 500 \
#     --restart_warmup_steps 50 \
#     --scheduler cosine_restarts \
#     --warmup_steps 400 \
#     --reset_optimizer_on_retff True \
#     --num_training_steps 40000 \
#     --save_every 3000 \
#     --eval_every 3000 \
#     --continue_from checkpoints/warmup4tff/model_5000 \
#     --tags relora_250M \
#     --exp_name tff_LRU_40k_k3_fast

# # Fourier
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc-per-node 1  torchrun_main.py \
#     --model_config configs/llama_130m.json \
#     --batch_size 2 \
#     --total_batch_size 4 \
#     --lr 1e-3 \
#     --max_length 512 \
#     --use_peft \
#     --retff 2 \
#     --cycle_length 2 \
#     --restart_warmup_steps 1 \
#     --scheduler cosine_restarts \
#     --warmup_steps 1 \
#     --reset_optimizer_on_retff True \
#     --num_training_steps 40000 \
#     --save_every 3000 \
#     --eval_every 3000 \
#     --tags relora_250M \
#     --exp_name debug_thread \