#! /bin/bash


export PYTHONPATH="$PYTHONPATH:/MEMENTO"
CUDA_DEVICES=5
now=$(date +%m-%d_%H-%M)

model_name=gpt-4o
job_name=tool_test


#### Stage 1 ####

echo "Running object-semantics-stage1 with ${model_name}"
type_name=object-semantics-stage1
run_name=${model_name}_${type_name}_${now}
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} HYDRA_FULL_ERROR=1 python -m src.main \
    --config-name pilot_study/pilot_object_stage1.yaml \
    hydra.job.name=${job_name} \
    hydra.run.dir=./outputs/${job_name}/${run_name} \

# echo "Running user-pattern-stage1 with ${model_name}"
# type_name=user-pattern-stage1-with-clean
# run_name=${model_name}_${type_name}_${now}
# CUDA_VISIBLE_DEVICES=4 HYDRA_FULL_ERROR=1 python -m src.main \
#     --config-name pilot_study/pilot_preference_stage1.yaml \
#     hydra.job.name=${job_name} \
#     hydra.run.dir=./outputs/${job_name}/${run_name} \


#### Stage 2 ####


# CUDA_VISIBLE_DEVICES=4 HYDRA_FULL_ERROR=1 python -m src.main \
#     --config-name pilot_study/pilot_object_stage2.yaml \
#     hydra.job.name=${job_name} \
#     hydra.run.dir=./outputs/${job_name}/${run_name} \


# CUDA_VISIBLE_DEVICES=4 HYDRA_FULL_ERROR=1 python -m src.main \
#     --config-name pilot_study/pilot_preference_stage2.yaml \
#     hydra.job.name=${job_name} \
#     hydra.run.dir=./outputs/${job_name}/${run_name} \

    
    
