#!/bin/bash

python src/preprocess/data_preprocess.py \
    --data_path data/datasets/PEAD/final_dataset_v0_unified.json \
    --output_dir data/datasets/PEAD/v0 \
    --parse_stage \
    --parse_type

# python src/preprocess/data_preprocess.py \
#     --data_path data/datasets/ours/pilot_org/preferences.json \
#     --output_dir data/datasets/ours/pilot \
#     --parse_stage 


