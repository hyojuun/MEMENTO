#!/bin/bash


export PYTHONPATH="$PYTHONPATH:/HabitatLLM"

now=$(date +%Y-%m-%d_%H-%M-%S)

model_name=gpt-4o
job_name=pilot_study
run_name=${model_name}_pilot_org_run3_randomize_wo_rearrange_${now}

CUDA_VISIBLE_DEVICES=4 HYDRA_FULL_ERROR=1 python -m src.main \
    --config-name main/pilot_org.yaml \
    hydra.job.name=${job_name} \
    hydra.run.dir=./outputs/${job_name}/${run_name} \



# job_name=pilot_study
# run_name=${model_name}_pilot_annotate_rag_top3_${now}

# CUDA_VISIBLE_DEVICES=4 HYDRA_FULL_ERROR=1 python -m src.main \
#     --config-name main/pilot_annotate_rag.yaml \
#     hydra.job.name=${job_name} \
#     hydra.run.dir=./outputs/${job_name}/${run_name} \


# job_name=pilot_study
# run_name=${model_name}_pilot_annotate_${now}

# CUDA_VISIBLE_DEVICES=4 HYDRA_FULL_ERROR=1 python -m src.main \
#     --config-name main/pilot_annotate.yaml \
#     hydra.job.name=${job_name} \
#     hydra.run.dir=./outputs/${job_name}/${run_name} \
    
    
    

# job_name=pilot_study
# run_name=${model_name}_pilot_org_filter

# CUDA_VISIBLE_DEVICES=4 HYDRA_FULL_ERROR=1 python -m src.main \
#     --config-name main/pilot_org_filter.yaml \
#     hydra.job.name=${job_name} \
#     hydra.run.dir=./outputs/${job_name}/${run_name} \