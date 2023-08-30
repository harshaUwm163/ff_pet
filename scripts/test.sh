CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc-per-node 1  torchrun_main.py \
    --model_config configs/llama_1b.json \
    --batch_size 1 \
    --total_batch_size 16 \
    --lr 5e-4 \
    --max_length 512 \
    --use_peft \
    --continue_from_peft 
    --retff 1000 \
    --cycle_length 1000 \
    --restart_warmup_steps 100 \
    --scheduler cosine_restarts \
    --warmup_steps 500 \
    --reset_optimizer_on_retff True \
    --num_training_steps 40000 \
    --save_every 3000 \
    --eval_every 3000 \
    --tags relora_350M \
    --exp_name debug_thread \