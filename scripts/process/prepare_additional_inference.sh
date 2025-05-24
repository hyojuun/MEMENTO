#! /bin/bash

export PYTHONPATH="$PYTHONPATH:/MEMENTO"

episode_id_list="586 436 516 704 408 531 508"

python src/postprocess/prepare_additional_data.py \
    --original_data_path /MEMENTO/data/datasets/PEAD/original/original_instructions.json \
    --episode_id_list $episode_id_list \
    --output_dir /MEMENTO/data/datasets/PEAD/original/additional \
    --name llama-3.1-8b-instruct_org_instruction_additionals