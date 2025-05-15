#! /bin/bash


export PYTHONPATH="$PYTHONPATH:/HabitatLLM"
# CUDA_DEVICES=6
CUDA_DEVICES=7
now=$(date +%m-%d_%H-%M)

model_name=gpt-4o
job_name=v0_study_new_ep5


#### Stage 1 ####


# echo "Running user-pattern-stage1 with ${model_name}"
# type_name=user-pattern-stage1
# run_name=${model_name}_${type_name}_${now}
# CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} HYDRA_FULL_ERROR=1 python -m src.main \
#     --config-name v0_experiment/v0_preference_stage1.yaml \
#     hydra.job.name=${job_name} \
#     hydra.run.dir=./outputs/${job_name}/${run_name} \


#### Stage 2 ####
echo "Running user-pattern-stage2 with ${model_name}"
type_name=user-pattern-stage2
run_name=${model_name}_${type_name}_${now}
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} HYDRA_FULL_ERROR=1 python -m src.main \
    --config-name v0_experiment/v0_preference_stage2.yaml \
    hydra.job.name=${job_name} \
    hydra.run.dir=./outputs/${job_name}/${run_name} \

# echo "Running user-pattern-stage2 with ${model_name}"
# type_name=user-pattern-stage2
# run_name=${model_name}_${type_name}_${now}
# CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} HYDRA_FULL_ERROR=1 python -m src.main \
#     --config-name v0_experiment/v0_preference_stage2.yaml \
#     hydra.job.name=${job_name} \
#     hydra.run.dir=./outputs/${job_name}/${run_name} \



    
    
