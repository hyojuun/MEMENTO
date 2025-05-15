#!bin/bash
A=6,7
B="meta-llama/Meta-Llama-3-8B-Instruct"
C=2 # length of A

CUDA_VISIBLE_DEVICES=$A
MODEL_NAME_OR_PATH=$B
TENSOR_PARALLEL_SIZE=$C

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES vllm serve $MODEL_NAME_OR_PATH \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --seed 42 \
    --port 8008 # Important 