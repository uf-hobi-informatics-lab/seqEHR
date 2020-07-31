#!/bin/bash

# training
python task.py \
  --model_path '../model' \
  --do_train \
  --learning_rate 1e-3 \
  --dropout_rate 0.05 \
  --train_epochs 50 \
  --hidden_dim 128 \
  --fc_dim 64 \
  --max_grad_norm 1.0 \

# test