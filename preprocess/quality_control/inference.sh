#!/bin/bash
export OPENAI_API_KEY="OPENAI_API_KEY"
# export OPENAI_ORGANIZATION=YOUR_ORGANIZATION_ID


python workspace/HabitatLLM/preprocess/quality_control/langchain_async_template.py \
    --model_name "gpt-4o" \
    --input_path /HabitatLLM/data/instructions/quality_input/user_pattern_val.json \
    --prompt /HabitatLLM/workspace/HabitatLLM/preprocess/quality_control/prompt.yaml \
    --prompt_key user_pattern \
    --save_instances \
    --save_dir /HabitatLLM/data/instructions/quality_output/user_pattern_val
