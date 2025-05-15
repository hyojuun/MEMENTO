import argparse
import os
import json
import gzip
from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DATA_STRUCTURE = {
    "config": None,
    "episodes": []
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/train.json.gz")
    parser.add_argument("--parse_stage", action="store_true", help="Parse stage")
    parser.add_argument("--parse_type", action="store_true", help="Parse type")
    parser.add_argument("--parse_per_scene", action="store_true", help="Parse per scene")
    parser.add_argument("--output_dir", type=str, default="data/train_processed")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    return args


def filter_data(data):
    # drop "used" key
    for episode in data["episodes"]:
        if "used" in episode:
            del episode["used"]
    return data

def load_data(data_path):
    if data_path.endswith(".gz"):
        with gzip.open(data_path, "rt") as f:
            data = json.load(f)
        
        # save in json file
        with open(data_path.replace(".gz", ""), "w") as f:
            json.dump(data, f, indent=4)
    else:
        with open(data_path, "r") as f:
            data = json.load(f)
    return data

def save_data(data, output_path):
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
        
    # save gzip file
    with gzip.open(output_path + ".gz", "wt") as f:
        json.dump(data, f, indent=4)
        
        
def get_episode_map(data):
    ## map stage1 and stage2 with related_episode_id
    ## csv file with two columns: stage1_episode_id, stage2_episode_id
    episode_map = {"stage1": [], "stage2": [], "org_episode_id": []}
    for episode in data["episodes"]:
        try:
            if int(episode["metadata"]["stage"]) == 1:
                episode_map["stage1"].append(episode["episode_id"])
                episode_map["stage2"].append(episode["metadata"]["related_episode_id"])
                episode_map["org_episode_id"].append(episode["original_data_info"]["episode_id"])
        except (ValueError, KeyError, TypeError) as e:
            print(f"Error: {e}")
            print(f"Episode ID: {episode['episode_id']}")
    return episode_map

def save_episode_map(episode_map, output_path):
    df = pd.DataFrame(episode_map)
    df.to_csv(output_path, index=False)
    



def parse_stage(data):
    stage_1_data = deepcopy(BASE_DATA_STRUCTURE)
    stage_2_data = deepcopy(BASE_DATA_STRUCTURE)
    
    for episode in data["episodes"]:
        try:
            if int(episode["metadata"]["stage"]) == 1:
                stage_1_data["episodes"].append(episode)
            else:
                stage_2_data["episodes"].append(episode)
        except (ValueError, KeyError, TypeError) as e:
            print(f"Error: {e}")
            print(f"Episode ID: {episode['episode_id']}")
    
    return stage_1_data, stage_2_data

    
def parse_type(data):
    object_semantics_type = deepcopy(BASE_DATA_STRUCTURE)
    user_pattern_type = deepcopy(BASE_DATA_STRUCTURE)
    
    for episode in data["episodes"]:
        if episode["metadata"]["episode_type"] == "object_semantics":
            object_semantics_type["episodes"].append(episode)
        else:
            user_pattern_type["episodes"].append(episode)
    
    return object_semantics_type, user_pattern_type

def merge_episodes(data):
    merged_data = deepcopy(BASE_DATA_STRUCTURE)
    for episode_list in data["episodes"]:
        for episode in episode_list:
            merged_data["episodes"].append(episode)
    return merged_data

## TODO: Scene 별로 task 나누기

def main(args):
    data = load_data(args.data_path)
    
    if type(data['episodes'][0]) == list and not args.parse_per_scene:
        data = merge_episodes(data)
    
    data = filter_data(data)
    print(f"Loaded {args.data_path}, total episodes: {len(data['episodes'])}")
    file_name = os.path.basename(args.data_path)
    file_name = file_name.split(".")[0]
    # get episode map
    episode_map = get_episode_map(data)
    save_episode_map(episode_map, os.path.join(args.output_dir, f"{file_name}_episode_map.csv"))
    
    # Handle case where both parse_stage and parse_type are True
    if args.parse_stage and args.parse_type:
        print("Parsing stage and type")
        # Parse stage first
        stage_1_data, stage_2_data = parse_stage(data)
        
        # Parse type within each stage
        stage_1_object_semantics, stage_1_user_pattern = parse_type(stage_1_data)
        stage_2_object_semantics, stage_2_user_pattern = parse_type(stage_2_data)
        
        # Save all combinations
        save_data(stage_1_object_semantics, os.path.join(args.output_dir, f"{file_name}_stage_1_object_semantics.json"))
        save_data(stage_1_user_pattern, os.path.join(args.output_dir, f"{file_name}_stage_1_user_pattern.json"))
        save_data(stage_2_object_semantics, os.path.join(args.output_dir, f"{file_name}_stage_2_object_semantics.json"))
        save_data(stage_2_user_pattern, os.path.join(args.output_dir, f"{file_name}_stage_2_user_pattern.json"))
    else:
        if args.parse_stage:
            print("Parsing stage")
            stage_1_data, stage_2_data = parse_stage(data)
            save_data(stage_1_data, os.path.join(args.output_dir, f"{file_name}_stage_1.json"))
            save_data(stage_2_data, os.path.join(args.output_dir, f"{file_name}_stage_2.json"))
        
        if args.parse_type:
            print("Parsing type")
            object_semantics_type, user_pattern_type = parse_type(data)
            save_data(object_semantics_type, os.path.join(args.output_dir, f"{file_name}_object_semantics.json"))
            save_data(user_pattern_type, os.path.join(args.output_dir, f"{file_name}_user_pattern.json"))

if __name__ == "__main__":
    args = parse_args()
    main(args)
    print("Done")