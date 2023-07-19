   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.run --nproc-per-node 6 torchrun_main.py \
      --model_config configs/llama_130m.json \
      --batch_size 75 \
      --total_batch_size 1200 \
      --lr 5e-4 \
      --max_length 512 \
      --tags warm_start_250M \
      --save_every 1000 \
      --num_training_steps 20000
