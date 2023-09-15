# script_path="$(realpath "$0")"
# CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.run --nproc-per-node 2 torchrun_main.py \
#    --model_config configs/llama_350m.json \
#    --batch_size 43 \
#    --total_batch_size 86 \
#    --lr 5e-4 \
#    --max_length 512 \
#    --tags warm_start_350M \
#    --save_every 5000 \
#    --eval_every 5000 \
#    --num_training_steps 40000 \
#    --script_path $script_path \
#    --exp_name time_350m_fullRank

script_path="$(realpath "$0")"
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.run --nproc-per-node 2  torchrun_main.py \
   --model_config configs/llama_350m.json \
   --batch_size 39 \
   --total_batch_size 78 \
   --lr 5e-4 \
   --max_length 512 \
   --use_peft \
   --retff 5000 \
   --cycle_length 5000 \
   --restart_warmup_steps 100 \
   --scheduler cosine_restarts \
   --warmup_steps 500 \
   --reset_optimizer_on_retff True \
   --num_training_steps 40000 \
   --save_every 5000 \
   --eval_every 5000 \
   --tags relora_350M \
   --script_path $script_path \
   --guide_after_n_restarts 20000000 \
   --scaling 1.0 \
   --num_frames 14 \
   --k_attn 64 \
   --l_attn 16 \
   --n_attn 1024 \
   --k_mlp 76 \
   --l_mlp 36 \
   --n_mlp 2736 \
   --exp_name time_350m_tff_attn_only \
   # --exp_name time_350m_tff_attn_mlp \
