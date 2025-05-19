#! /bin/bash

python src/postprocess/analyze.py \
    --original_path /HabitatLLM/data/datasets/PEAD/v1/v1_val_stage1.json \
    --anal_path /HabitatLLM/data/analysis/temp_for_check.json \
    --output_path /HabitatLLM/data/outputs/gpt-4o_top_5_wo_gold/gpt-4o_v1_stage2_04-10_13-55/results/episode_result_log.csv \
    --memory_path /HabitatLLM/data/memory/v1_experiment/gpt-4o_v1_stage1_04-10_01-45/episode_result_log.csv