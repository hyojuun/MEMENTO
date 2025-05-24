#! /bin/bash

export PYTHONPATH="$PYTHONPATH:/MEMENTO"
CUDA_DEVICES=0 ####### SHOULD CHANGE

now=$(date +%m-%d_%H-%M)

model_name=gpt-4o ####### SHOULD CHANGE
job=stage1_top_k_5 ####### SHOULD CHANGE
job_name=${model_name}_${job}

#### Stage 1 ####

echo "Running acquisition stage with ${job_name}"
data_type_name=acquisition_stage

run_name=${model_name}_${data_type_name}_${now}
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} HYDRA_FULL_ERROR=1 python -m src.main \
    --config-name v1_experiment/acquisition_stage.yaml \
    hydra.job.name=${job_name} \
    hydra.run.dir=./outputs/${job_name}/${run_name} \