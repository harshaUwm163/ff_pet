# script_path="$(realpath "$0")"
# CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.run --nproc-per-node 2 torchrun_main.py \
#    --model_config configs/llama_250m.json \
#    --batch_size 53 \
#    --total_batch_size 106 \
#    --lr 5e-4 \
#    --max_length 512 \
#    --tags warm_start_250M \
#    --save_every 5000 \
#    --eval_every 5000 \
#    --num_training_steps 40000 \
#    --script_path $script_path \
#    --exp_name time_250m_fullRank
#    # --exp_name warmup4tff_250m

script_path="$(realpath "$0")"
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.run --nproc-per-node 2  torchrun_main.py \
   --model_config configs/llama_250m.json \
   --batch_size 42 \
   --total_batch_size 84 \
   --lr 5e-4 \
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
   --script_path $script_path \
   --guide_after_n_restarts 20000000 \
   --scaling 1.0 \
   --num_frames 15 \
   --k_attn 48 \
   --l_attn 16 \
   --n_attn 768 \
   --k_mlp 80 \
   --l_mlp 32 \
   --n_mlp 2560 \
   --exp_name time_250m_tff_attn_mlp \
   # --exp_name time_130m_tff_attn_only \

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
