import json
from copy import deepcopy

def output_as_dataset(episodes: list, output_path: str):
    dataset = {
        "config": None,
        "episodes": episodes
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    
    print(f"Dataset saved to {output_path}")


def extract_original_info(episodes: list, data_path: str):
    # 1. original info 더하기
    for i, episode in enumerate(episodes):
        episode_id = episode['episode_id']
        instruction = episode['instruction']
        data_path = data_path
        episodes[i]['original_data_info'] = {
            "episode_id": episode_id, 
            "instruction": instruction, 
            "data_path": data_path
        }
    
    return episodes

# def renew_episode_info(episodes: list, start_idx: int):
#     reordered = episodes[5:10] + episodes[0:5]
#     for i, episode in enumerate(reordered):
#         episode['episode_id'] = str(start_idx + i)
#     episodes[:] = reordered
#     return episodes


# def add_metadata(episodes: list):
#     # 3. metadata 더하기
#     for i, episode in enumerate(episodes):
#         if i >= 5:
#             episode_type = 'user_pattern'
#         else:
#             episode_type = 'object_semantics'
            
#         related_episode_id = str(int(episode['episode_id']) + 1000)
#         episodes[i]['metadata'] = {
#             'stage': "1",
#             'related_episode_id': related_episode_id,
#             'episode_type': episode_type
#         }
    
#     return episodes

# def add_stage_2(episodes: list):
#     stage_2 = []
    
#     for episode in episodes:
#         new_episode = deepcopy(episode) 
#         new_episode['metadata'] = {
#             'stage': "2",
#             'related_episode_id': str(episode['episode_id']),
#             'episode_type': episode['metadata']['episode_type']
#         }
#         new_episode['episode_id'] = str(int(episode['episode_id']) + 1000)
#         stage_2.append(new_episode)
    
#     episodes.extend(stage_2)
    
#     return episodes
    
# ========================================================
# utils for post_pipeline_v2.py

def renew_episode_info(episodes: list, start_idx: int):
    for i, episode in enumerate(episodes):
        episode['episode_id'] = str(start_idx + i)
    return episodes


def add_metadata(episodes: list, split: str, idx: int):
    # 3. metadata 더하기
    for i, episode in enumerate(episodes):
        if idx == 0:
            episode_type = 'user_pattern'
        elif idx == 1:
            episode_type = 'object_semantics'
            
        related_episode_id = str(int(episode['episode_id']) + 2000)
        episodes[i]['metadata'] = {
            'stage': "1",
            'related_episode_id': related_episode_id,
            'episode_type': episode_type,
            'source_file': split
        }
    
    return episodes

def add_stage_2(episodes: list):
    stage_2 = []
    
    for episode in episodes:
        new_episode = deepcopy(episode) 
        new_episode['metadata'] = {
            'stage': "2",
            'related_episode_id': str(episode['episode_id']),
            'episode_type': episode['metadata']['episode_type']
        }
        new_episode['episode_id'] = str(int(episode['episode_id']) + 2000)
        stage_2.append(new_episode)
    
    episodes.extend(stage_2)
    
    return episodes
