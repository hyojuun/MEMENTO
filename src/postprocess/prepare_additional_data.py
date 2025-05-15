import argparse
import os
import json
import gzip
import pandas as pd
import numpy as np
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="Name of the additional data to prepare")
    parser.add_argument("--original_data_path", type=str, help="Path to the original data")
    parser.add_argument("--episode_id_list", nargs='+', type=int, help="List of episode IDs to process")
    parser.add_argument("--output_dir", type=str, default="data/additional_data/")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    return args


def prepare_additional_data(args, original_data):
    new_data = {
        "config": None,
        "episodes": [],
    }

    for original_episode in original_data["episodes"]:
        if int(original_episode["episode_id"]) not in args.episode_id_list:
            continue

        new_data["episodes"].append(original_episode)

    return new_data


def main(args):
    # print list
    print(f"Episode type: {type(args.episode_id_list[0])}")
    print(f"Episode IDs: {args.episode_id_list}")
    
    # load original data
    with open(args.original_data_path, "r") as f:
        original_data = json.load(f)

    
    new_data = prepare_additional_data(args, original_data)

    # save the new data
    gzip_path = os.path.join(args.output_dir, f"{args.name}.json.gz")
    print(f"Saving the new data to {gzip_path}")
    with gzip.open(gzip_path, "wt") as f:
        json.dump(new_data, f)
    
    # save in json
    with open(os.path.join(args.output_dir, f"{args.name}.json"), "w") as f:
        json.dump(new_data, f, indent=4)
    

if __name__ == "__main__":
    args = get_args()
    main(args)


