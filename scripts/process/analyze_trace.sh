#! /bin/bash

job_name="gpt-4o_combined_episode"
run_name="gpt-4o_v1_stage1_04-22_02-08"
dataset_name="v1_val_stage1_filtered_combined.json.gz"

# 1. 만약, job_name / run_name 따로 하고 싶다 -> 아래처럼 full 넣어주면 됨
# --anal_folder_path /MEMENTO/outputs/${job_name}/${run_name}/results/${dataset_name}/planner-log

# 2. 만약, job_name 같은 모든 것 불러오고 싶다 -> job_name 까지만 넣어주기
# --anal_folder_path /MEMENTO/outputs/${job_name}

python src/postprocess/trace_analyze.py \
    --original_path /MEMENTO/data/datasets/PEAD/v1/v1_val_stage1_filtered.json \
    --anal_folder_path /MEMENTO/outputs/${job_name}/${run_name}/results/${dataset_name}/planner-log \
    --metadata_path /MEMENTO/data/datasets/PEAD/v1/episode_metadata.csv \
    --output_path trace_analyze/${job_name}/trace_analyze.json \

# --anal_folder_path /MEMENTO/outputs/${job_name} \