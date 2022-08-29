#!/bin/bash

set -e
set -u

DST_DIR="/mnt/inspurfs/user-fs/zhaoyu/GlyphBERT"
cd "$DST_DIR"
pwd

python classification.py \
  --device=1 \
  --batch_size=32 \
  --batch_expand_times=1 \
  --epoch=20 \
  --warm_up=0.1 \
  --lr=3e-5 \
  --weight_decay=0.01 \
  --dataset_name="onlinesenti_cls" \
  --steps_eval=0.2 \
  --start_eval_epoch=3 \
  --pretrained_model_path="/mnt/inspurfs/user-fs/zhaoyu/GlyphBERT/save/AddBertResPos3-epoch4-loss1.25952.pt" \

# --paq_qas_path="/raid/zy/PAQ/data/paq/TQA_TRAIN_NQ_TRAIN_PAQ/tqa-train-nq-train-PAQ.jsonl" \
# --paq_keys_dir="/raid/zy/Multi-PAQ-main/scripts/repaq_models/vectors/multi_base_256_vectors" \
#  --paq_qas_path="/raid/zy/PAQ/data/paq/TQA_TRAIN_NQ_TRAIN_PAQ/TQA_TRAIN_NQ_TRAIN_PAQ_1300672.jsonl" \
#  --paq_keys_dir="/raid/zy/Multi-PAQ-main/scripts/repaq_models/vectors/multi_base_256_vectors_debug" \
