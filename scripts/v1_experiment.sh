#! /bin/bash

export PYTHONPATH="$PYTHONPATH:/HabitatLLM"
CUDA_DEVICES=0,1,2,3,4,5,6,7
now=$(date +%m-%d_%H-%M)

model_name=gpt-4o # what model?
job=valid_set_top_5 # what job?
job_name=${model_name}_${job}

#### Stage 1 ####

echo "Running v1_stage1 with ${model_name}"
type_name=v1_stage1
run_name=${model_name}_${type_name}_${now}
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} HYDRA_FULL_ERROR=1 python -m src.main \
    --config-name v1_experiment/v1_stage1.yaml \
    hydra.job.name=${job_name} \
    hydra.run.dir=./outputs/${job_name}/${run_name} \

#### Stage 2 ####

echo "Running v1_stage2 with ${model_name}"
type_name=v1_stage2
run_name=${model_name}_${type_name}_${now}
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} HYDRA_FULL_ERROR=1 python -m src.main \
    --config-name v1_experiment/v1_stage2.yaml \
    hydra.job.name=${job_name} \
    hydra.run.dir=./outputs/${job_name}/${run_name} \