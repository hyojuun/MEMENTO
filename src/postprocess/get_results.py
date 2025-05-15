import argparse
import os
import pandas as pd
import json



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--original_data_path", type=str, default=None, help="Path to the original data")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    return args

def read_csv(path):
    return pd.read_csv(path)

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def check_missing_episodes(org_data, episode_results):
    missing_episodes = []
    for episode in org_data:
        if episode not in episode_results:
            missing_episodes.append(episode)
    return missing_episodes

def main(args):
    
    ## Read original data
    org_data = read_json(args.original_data_path)

    ## Read episode results
    episode_results_path = os.path.join(args.results_dir, "episode_result_log.csv")
    episode_results = read_csv(episode_results_path)
    
    ## Check missing episodes
    missing_episodes = check_missing_episodes(org_data, episode_results)
    print(f"Missing episodes: {missing_episodes}")

if __name__ == "__main__":
    args = get_args()
    main(args)