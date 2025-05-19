#! /bin/bash


## 1 step
# python src/postprocess/get_detailed_results.py \
#     --stage 1 \
#     --type "all" \
#     --result_path "/HabitatLLM/data/outputs/Llama-3.1-8B-instruct_build_memory/Llama-3.1-8B-instruct_v1_stage1_04-14_07-47/results/episode_result_log.csv" \
#     --save_dir "/HabitatLLM/results/llama-3.1-8b-instruct/stage1_wo_memory"

# python src/postprocess/get_detailed_results.py \
#     --stage 2 \
#     --type "all" \
#     --result_path "/HabitatLLM/data/outputs/gpt-4o_top_5_wo_gold/gpt-4o_v1_stage2_04-10_13-55/results/episode_result_log.csv" \
#     --save_dir "/HabitatLLM/results/gpt-4o/top_5_wo_gold"

## 2 step
# python src/postprocess/get_detailed_results.py \
#     --stage 2 \
#     --type "all" \
#     --do_integrate \
#     --integration_path_list "/HabitatLLM/data/outputs/gpt-4o_top_1/gpt-4o_v1_stage2_04-10_13-17/results/episode_result_log.csv" \
#     "/HabitatLLM/data/outputs/gpt-4o_top_1/gpt-4o_v1_stage2_04-11_05-11/results/episode_result_log.csv" \
#     --save_dir "/HabitatLLM/results/gpt-4o/top_1"

## 3 step
python src/postprocess/get_detailed_results.py \
    --stage 3 \
    --type "all" \
    --result_path "/HabitatLLM/outputs/gpt-4o_wo_memory/gpt-4o_v1_stage3_05-02_08-31/results/episode_result_log.csv" \
    --save_dir "/HabitatLLM/results/gpt-4o/stage3_wo_memory"

    # /HabitatLLM/outputs/llama-8b_wo_memory/llama-8b_v1_stage3_05-02_08-30