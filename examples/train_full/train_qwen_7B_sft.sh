#!/bin/bash
if[ -n "$MLP_WORKER_NUM"]; then
  NNODES="$MLP_WORKER_NUM"
  GPUS_PER_NODE=8
else
  NNODES=1
  GPUS_PER_NODE=1
fi
if [ -n "$MLP_ROLE_INDEX"]; then
  NODE_RANK="$MLP_ROLE_INDEX"
else
  NODE_RANK=0
fi
if [ -n "$MLP_WORKER_0_HOST"]; then
  MASTER_ADDR="$MLP_WORKER_0_HOST"
  MASTER_PORT="$MLP_WORKER_0_PORT"
else
  MASTER_ADDR=localhost
  MASTER_PORT=12345
fi

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    scr/train.py \
    --model_name_or_path /home/xxx/models/Qwen2-7B-Instruct \
    --tokenizer_name_or_path /home/xxx/models/Qwen2-7B-Instruct \
    --stage sft \
    --do_train true \
    --finetuning_type full \
    --pref_beta 0.1 \
    --pref_loss sigmoid \
    --cutoff_len 2048 \
    --deepspeed examples/deepspeed/ds_z3_offload_config.json \
    --dataset xxx,xxx,xxx \
    --template qwen \
    --max_samples 10000000000 \
    --overwrite_cache true \
    --preprocessing_num_workers 64 \
    --output_dir /home/xxx/out_model/qwen2_7b_sft/sft_qwen_7b_lr1e6 \
    --logging_steps 10 \
    --save_strategy epoch \
    --plot_loss true \
    --overwrite_output_dir true \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1.0e-6 \
    --num_train_epoch 5.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 true \
    --ddp_timeout 360000000 \
    --flash_attn fa2






