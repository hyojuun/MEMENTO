import glob
import os
import re
import shutil
import json

from src.agent.env.dataset import MemoryManagementDatasetV0


def episode_scene_map(dataset: MemoryManagementDatasetV0):
    ep_scene_map_dict = {}
    
    for episode in dataset.episodes:
        ep_scene_map_dict[episode.episode_id] = episode.scene_id
    
    return ep_scene_map_dict


def build_memory(output_path: str, data_path: str, dataset: MemoryManagementDatasetV0) -> str:
    """
    Builds the memory path for a given job name and run name.
    Args:
        output_path (str): The name of the run.
        data_path (str): The base path for the data.    
    Returns:
        str: The constructed memory path.
    """
    # /HabitatLLM/outputs/v1_experiment/gpt-4o_v1_stage1_04-09_15-59/results
    # 여기서 v1_experiment/gpt-4o_v1_stage1_04-09_15-59/results를 파싱해야 함
    job_name = output_path.split('/')[-3] # v1_experiment
    run_name = output_path.split('/')[-2] # gpt-4o_v1_stage1_04-09_15-59

    # data/datasets/PEAD/v1/filtered_object_semantics_1_val.json.gz
    # dataset_name = data_path.split('/')[-1]
    
    # /HabitatLLM/outputs/{job_name}/{run_name}/results/{dataset_name}/prompts/0/prompt-episode_{episode_id}_0-0.txt
    
    # 그렇다면, episode_id 별로 돌아야 함.
    # 그러려면, 해당 경로의 파일들을 옮겨야 함.
    # 옮길 대상은
    # memory/v1_experiment/gpt-4o_v1_stage1_04-09_15-59/traces
    
    # !!!!!!!!!!RAG에서 고칠것!!!!!!!!!!
    #     self.rag = RAG(
    #     plan_config.example_type,
    #     plan_config.rag_dataset_dir,
    #     plan_config.rag_data_source_name,
    #     plan_config.llm,
    # )
    dataset_name = data_path.split('/')[-1]
    prompt_files = glob.glob(f"{output_path}/{dataset_name}/prompts/0/prompt-episode_*.txt", recursive=True)
    trace_files = glob.glob(f"{output_path}/{dataset_name}/traces/0/trace-episode_*.txt", recursive=True)
    
    ep_scene_map_dict = episode_scene_map(dataset) # episode_id -> scene_id
    base_memory_path = f"/HabitatLLM/memory/{job_name}/{run_name}"
    os.makedirs(base_memory_path, exist_ok=True)
    
    # mapper_path = f"/HabitatLLM/memory/{job_name}/mapper.json"
    # if not os.path.exists(mapper_path):
    #     with open(mapper_path, 'w') as mapper_file:
    #         json.dump({"learned": None, "oracle": None}, mapper_file)
    
    shutil.copy(f"{output_path}/episode_result_log.csv", os.path.join(base_memory_path, "episode_result_log.csv"))
    
    for file_path in prompt_files:
        file_name = file_path.split('/')[-1]
        
        match = re.search(r"prompt-episode_(\d+)_\d+-0.txt", file_name)
        
        if match:
            episode_id = match.group(1)
            scene_id = ep_scene_map_dict.get(episode_id)
            
            os.makedirs(os.path.join(base_memory_path, scene_id, 'prompts', '0'), exist_ok=True)
            shutil.copy(file_path, os.path.join(base_memory_path, scene_id, 'prompts', '0', file_name))
    
    for file_path in trace_files:
        file_name = file_path.split('/')[-1]
        
        match = re.search(r"trace-episode_(\d+)_\d+-0.txt", file_name)
        
        if match:
            episode_id = match.group(1)
            scene_id = ep_scene_map_dict.get(episode_id)
            
            os.makedirs(os.path.join(base_memory_path, scene_id, 'traces', '0'), exist_ok=True)
            shutil.copy(file_path, os.path.join(base_memory_path, scene_id, 'traces', '0', file_name))
            # memory/{job_name}/{run_name}/{scene_id}/prompt-episode_{episode_id}_0-0.txt
            
            # if scene_id:
            #     # Create the destination directory if it doesn't exist
            #     destination_dir = os.path.join(output_path, "traces", scene_id)
            #     os.makedirs(destination_dir, exist_ok=True)
                
            #     # Move the file to the new location
            #     destination_path = os.path.join(destination_dir, file_name)
            #     os.rename(file_path, destination_path)
        
        #file[-1] "prompt-episode_*_0-0.txt"
    
    
    # with open(mapper_path, 'r') as mapper_file:
    #     mapper_data = json.load(mapper_file)  

    # mapper_data["learned"] = run_name  

    # with open(mapper_path, 'w') as mapper_file:
    #     json.dump(mapper_data, mapper_file, indent=4)
    
    
    
    # 1. load_data_llm -> state_success인 것만 불러오는 것 => parameter로 조정할 수 있도록 하기
    # 2. 

    # learned_memory
    # gold_memory
    # => 적을 수 있음