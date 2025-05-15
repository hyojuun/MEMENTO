import argparse
import json
import os
from glob import glob
from collections import defaultdict
from typing import Dict, List, Set

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 뭘 넣어줄까?
# outputs/llama-3.1-70b-instruct_stage1/~/results/~/planner-log
def get_args():
    parser = argparse.ArgumentParser(description='Process some data files.')
    parser.add_argument('--original_path', type=str, required=True, help='Path to the original dataset path')
    parser.add_argument('--anal_folder_path', type=str, required=True, help='Path to the analysis folder path')
    parser.add_argument('--metadata_path', type=str, required=True, help='Path to the dataset metadata csv file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output CSV file')
    args = parser.parse_args()
    return args


def load_data(anal_folder_path, metadata_path, recursive):
    all_analysis = {
        'traces': []
    }
    
    if recursive:
        search_pattern = os.path.join(anal_folder_path, "**/planner-log/planner-log-episode_*.json")
        planner_log_files = glob(search_pattern, recursive=True)
        
        search_pattern_2 = os.path.join(anal_folder_path, "**/results/episode_result_log.csv")
        episode_result_log_files = glob(search_pattern_2, recursive=True)
        df_list = []
        for file in episode_result_log_files:
            df = pd.read_csv(file)
            df_list.append(df)
        episode_result_log_df = pd.concat(df_list, ignore_index=True)
        
    else:
        planner_log_files = glob(os.path.join(anal_folder_path, "*"))
        episode_result_log_df = pd.read_csv(os.path.join('/'.join(anal_folder_path.split('/')[:-2]), "episode_result_log.csv"))
    
    metadata_df = pd.read_csv(metadata_path)
    
    
    for file in planner_log_files:
        episode_id = int(os.path.basename(file).split('.')[0].replace('.json', '').split('_')[1])
        
        with open(file, "r") as f:
            raw_data = json.load(f)
        
        episode_trace = {
            'episode_id': episode_id,
            'episode_type': metadata_df[metadata_df['episode_id'] == episode_id]['episode_type'].values[0],
            'subtype': metadata_df[metadata_df['episode_id'] == episode_id]['subtype'].values[0],
            'task': raw_data['task'],
            'task_percent_complete': episode_result_log_df[episode_result_log_df['episode_id'] == episode_id]['task_percent_complete'].values[0],
            'task_state_success': episode_result_log_df[episode_result_log_df['episode_id'] == episode_id]['task_state_success'].values[0],
            'final_evaluation': raw_data['steps'][-1]['stats']['task_explanation'],
            'thoughts': [],
            'actions': [],
        }
        
        data = raw_data['steps']
        current_action = None
        for i, log in enumerate(data):
            try:
                if log.get('thought').get("0"):
                    # 마지막엔 agent_collisions, agent_positions, agent_states이 없음. 따라서, 예외처리 필요함
                    episode_trace['thoughts'].append({
                        'sim_step_count': log['sim_step_count'],
                        'replanning_count': log['replanning_count']["0"],
                        'agent_states': log['agent_states']["0"],
                        'thought': log['thought']["0"]
                        })
                if log['high_level_actions']["0"] != current_action:
                    episode_trace['actions'].append({
                        'sim_step_count': log['sim_step_count'],
                        'replanning_count': log['replanning_count']["0"],
                        'agent_states': log['agent_states']["0"],
                        'action': log['high_level_actions']["0"][0],
                        'task_percent_complete': data[i+1]['stats']['task_percent_complete'],
                        })
                    current_action = log['high_level_actions']["0"]
            except Exception as e:
                continue
        
        all_analysis['traces'].append(episode_trace)
    
    
    return all_analysis, metadata_df, episode_result_log_df

def analyze_replanning_actions(all_analysis, metadata_df, episode_result_log_df):
    episode_actions = {}
    
    for trace in all_analysis['traces']:
        episode_id = trace['episode_id']
        episode_actions[episode_id] = []
        
        for action_data in trace['actions']:
            if action_data['replanning_count'] > 0:
                episode_actions[episode_id].append(action_data['action'])
    
    action_ratios = {}
    for episode_id, actions in episode_actions.items():
        if not actions:
            continue
            
        total = len(actions)
        ratios = {}
        for action in set(actions):
            action_count = actions.count(action)
            ratios[action] = action_count
        
        action_ratios[episode_id] = {
            "task_percent_complete": episode_result_log_df[episode_result_log_df['episode_id'] == episode_id]['task_percent_complete'].values[0],
            "task_state_success": episode_result_log_df[episode_result_log_df['episode_id'] == episode_id]['task_state_success'].values[0],
            "episode_type": metadata_df[metadata_df['episode_id'] == episode_id]['episode_type'].values[0],
            "subtype": metadata_df[metadata_df['episode_id'] == episode_id]['subtype'].values[0],
            "ratios": ratios
        }
        
    return action_ratios

def analyze_replanning_steps(all_analysis):
    action_steps = {}
    episode_max_replans = {}
    
    for trace in all_analysis['traces']:
        episode_id = trace['episode_id']
        max_replan = 0
        
        for action in trace['actions']:
            max_replan = max(max_replan, action['replanning_count'])
        
        if max_replan > 0: 
            episode_max_replans[episode_id] = max_replan
    
    for trace in all_analysis['traces']:
        episode_id = trace['episode_id']
        if episode_id not in episode_max_replans:
            continue
            
        max_replan = episode_max_replans[episode_id]
        
        for action_data in trace['actions']:
            if action_data['replanning_count'] > 0:
                action = action_data['action']
                
                normalized_step = action_data['replanning_count'] / max_replan
                
                if action not in action_steps:
                    action_steps[action] = []
                
                action_steps[action].append(normalized_step)
    
    stats = {}
    for action, steps in action_steps.items():
        stats[action] = {
            'mean': np.mean(steps),
            'std': np.std(steps),
            'median': np.median(steps),
            'min': min(steps),
            'max': max(steps),
            'count': len(steps)
        }
        
    return stats

def analyze_sim_steps(all_analysis):
    action_sim_steps = {}
    episode_max_steps = {}
    
    for trace in all_analysis['traces']:
        episode_id = trace['episode_id']
        max_steps = 0
        
        for action in trace['actions']:
            max_steps = max(max_steps, action['sim_step_count'])
            
        if max_steps > 0: 
            episode_max_steps[episode_id] = max_steps
    
    for trace in all_analysis['traces']:
        episode_id = trace['episode_id']
        if episode_id not in episode_max_steps:
            continue
            
        max_steps = episode_max_steps[episode_id]
        
        for action_data in trace['actions']:
            if action_data['replanning_count'] > 0:
                action = action_data['action']
                
                normalized_step = action_data['sim_step_count'] / max_steps
                
                if action not in action_sim_steps:
                    action_sim_steps[action] = []
                
                action_sim_steps[action].append(normalized_step)
    
    stats = {}
    for action, steps in action_sim_steps.items():
        stats[action] = {
            'mean': np.mean(steps),
            'std': np.std(steps),
            'median': np.median(steps),
            'min': min(steps),
            'max': max(steps),
            'count': len(steps),
            'raw_mean': np.mean([s * episode_max_steps.get(trace['episode_id'], 1) 
                               for s, trace in zip(steps, all_analysis['traces'])])
        }
        
    return stats


def plot_results(all_analysis, metadata_df, episode_result_log_df):
    action_ratios = analyze_replanning_actions(all_analysis, metadata_df, episode_result_log_df)
    step_stats = analyze_replanning_steps(all_analysis)
    sim_step_stats = analyze_sim_steps(all_analysis)
    
    return {
        'action_ratios': action_ratios,
        'step_stats': step_stats,
        'sim_step_stats': sim_step_stats,
    }


def main(args):
    if args.anal_folder_path.endswith('planner-log'):
        all_analysis, metadata_df, episode_result_log_df = load_data(args.anal_folder_path, args.metadata_path, recursive=False)
    else:
        all_analysis, metadata_df, episode_result_log_df = load_data(args.anal_folder_path, args.metadata_path, recursive=True)
    
    results = plot_results(all_analysis, metadata_df, episode_result_log_df)
    
    output_dir = os.path.dirname(args.output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output_path, 'w') as f:
        json.dump(all_analysis, f, indent=2)
    
    with open(os.path.join(output_dir, 'analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Analysis complete. Results saved to {args.output_path}")

if __name__ == "__main__":
    args = get_args()
    main(args)
