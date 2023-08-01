# CUDA_VISIBLE_DEVICES=1,2,3,4,5 python -m torch.distributed.run --nproc-per-node 5 torchrun_main.py \
#    --model_config configs/llama_350m.json \
#    --batch_size 16 \
#    --total_batch_size 160 \
#    --lr 5e-4 \
#    --max_length 512 \
#    --tags warm_start_350M \
#    --save_every 1000 \
#    --num_training_steps 20000 \
#    --exp_name warmup4tff_350m_5gpus

# Olvi 2
CUDA_VISIBLE_DEVICES=1,2,3,4,5 python -m torch.distributed.run --nproc-per-node 5  torchrun_main.py \
    --model_config configs/llama_350m.json \
    --batch_size 16 \
    --total_batch_size 160 \
    --lr 1e-3 \
    --max_length 512 \
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
    # --continue_from checkpoints/warmup4tff_250m_2/model_5000 \
    --tags relora_350M \
    --exp_name tff_350m_20k \

# # Fourier
#     --retff 20 \
#     --cycle_length 20 \
#     --restart_warmup_steps 10 \
#     --scheduler cosine_restarts \
#     --warmup_steps 10 \
#     --reset_optimizer_on_retff True \
#     --num_training_steps 20000 \
#     --save_every 30 \
#     --eval_every 30 \
#     --tags relora_250M \
#     --exp_name debug_thread 

# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.run --nproc-per-node 6  torchrun_main.py \
#     --model_config configs/llama_130m.json \
#     --batch_size 75 \
#     --total_batch_size 900 \
#     --lr 1e-3 \
#     --max_length 512 \
#     --use_peft \
#     --retff 1000 \
#     --cycle_length 1000 \
#     --restart_warmup_steps 100 \
#     --scheduler cosine_restarts \
#     --warmup_steps 500 \
#     --reset_optimizer_on_retff True \
#     --num_training_steps 40000 \
#     --save_every 3000 \
#     --eval_every 3000 \
#     --continue_from checkpoints/warmup4tff/model_5000 \
#     --tags relora_250M \
#     --exp_name tff_LRU_40k

# Fourier
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
