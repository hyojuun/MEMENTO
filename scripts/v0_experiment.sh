#! /bin/bash


export PYTHONPATH="$PYTHONPATH:/HabitatLLM"
# CUDA_DEVICES=4
CUDA_DEVICES=0
now=$(date +%m-%d_%H-%M)

model_name=gpt-4o
job_name=v0_study_new_ep5


#### Stage 1 ####

echo "Running object-semantics-stage1 with ${model_name}"
type_name=object-semantics-stage1
run_name=${model_name}_${type_name}_${now}
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} HYDRA_FULL_ERROR=1 python -m src.main \
    --config-name v0_experiment/v0_object_stage1.yaml \
    hydra.job.name=${job_name} \
    hydra.run.dir=./outputs/${job_name}/${run_name} \

# echo "Running object-semantics-stage2 with ${model_name}"
# type_name=object-semantics-stage2
# run_name=${model_name}_${type_name}_${now}
# CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} HYDRA_FULL_ERROR=1 python -m src.main \
#     --config-name v0_experiment/v0_object_stage2.yaml \
#     hydra.job.name=${job_name} \
#     hydra.run.dir=./outputs/${job_name}/${run_name} \



#### Stage 2 ####

echo "Running object-semantics-stage2 with ${model_name}"
type_name=object-semantics-stage2
run_name=${model_name}_${type_name}_${now}
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} HYDRA_FULL_ERROR=1 python -m src.main \
    --config-name v0_experiment/v0_object_stage2.yaml \
    hydra.job.name=${job_name} \
    hydra.run.dir=./outputs/${job_name}/${run_name} \



    
    
