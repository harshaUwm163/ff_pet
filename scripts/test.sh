# script_path="$(realpath "$0")"
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.run --nproc-per-node 1  torchrun_main.py \
#     --model_config configs/llama_250m.json \
#     --batch_size 1 \
#     --total_batch_size 1 \
#     --lr 5e-4 \
#     --max_length 1024 \
#     --use_peft \
#     --retff 4 \
#     --cycle_length 4 \
#     --restart_warmup_steps 1 \
#     --scheduler cosine_restarts \
#     --warmup_steps 1 \
#     --reset_optimizer_on_retff True \
#     --num_training_steps 40000 \
#     --save_every 2000 \
#     --eval_every 2000 \
#     --tags relora_1b \
#     --exp_name debug_thread \
#     --script_path $script_path \
#     --scaling 4.0
