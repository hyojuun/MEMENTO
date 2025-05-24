#! /bin/bash

export PYTHONPATH="$PYTHONPATH:/MEMENTO"
CUDA_DEVICES=0 ####### SHOULD CHANGE

now=$(date +%m-%d_%H-%M)

model_name=gpt-4o-mini ####### SHOULD CHANGE
job=build_memory ####### SHOULD CHANGE
job_name=${model_name}_${job}

#### Stage 1 ####

echo "Running v1_stage1 with ${job_name}"
data_type_name=v1_stage1

run_name=${model_name}_${data_type_name}_${now}
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} HYDRA_FULL_ERROR=1 python -m src.main \
    --config-name v1_experiment/v1_stage1.yaml \
    hydra.job.name=${job_name} \
    hydra.run.dir=./outputs/${job_name}/${run_name} \