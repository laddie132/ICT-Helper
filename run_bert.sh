#!/usr/bin/env bash

# shellcheck disable=SC2164
cd bert

#export LD_LIBRARY_PATH="/usr/local/cuda/lib64":$LD_LIBRARY_PATH
export BERT_BASE_DIR=/home/lh/data-emb/chinese_L-12_H-768_A-12
export DATA_DIR=/home/lh/ICT-Helper/data
export OUTPUT_DIR=/home/lh/ICT-Helper/outputs/bert_异常类型
export TASK_NAME=ict_et
#export OUTPUT_DIR=/home/lh/ICT-Helper/outputs/bert_故障组件
#export TASK_NAME=ict_fc
#export OUTPUT_DIR=/home/lh/ICT-Helper/outputs/bert_形成原因
#export TASK_NAME=ict_rs

CUDA_VISIBLE_DEVICES=2 python run_classifier.py \
  --task_name=$TASK_NAME \
  --do_train=true \
  --do_eval=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR