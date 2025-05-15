#! /bin/bash

python workspace/HabitatLLM/preprocess/quality_control/parse.py \
    --input_path /HabitatLLM/data/instructions/quality_output/user_pattern_val_total_results.json \
    --output_path /HabitatLLM/data/instructions/quality_output/user_pattern_val_total_results_parsed.json

