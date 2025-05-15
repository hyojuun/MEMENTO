import json
from collections import defaultdict, Counter
from typing import List, Dict, Callable
import random
import itertools
from tqdm import tqdm

from utils import output_as_dataset, renew_episode_info, add_metadata, extract_original_info, add_stage_2
import pandas as pd


def check_used_object(df: pd.DataFrame, obj_id: str):
    return df[df['id'] == obj_id]

def check_used_object_category(df: pd.DataFrame, obj_id: str, category: str) -> bool:
    """
    just for object_category_df
    """
    filtered = df[(df['id'] == obj_id) & (df['clean_category'] == category)]
    return not filtered.empty 
    

def check_used_object_category_from_eval(eval: Dict, df: pd.DataFrame, keys: list):
    if eval['function_name'] in ["is_on_top", "is_inside", "is_in_room"]:
        handle = eval['args']['object_handles'][0]
        obj_id = handle.split(":")[0][:-1]
        target_df = check_used_object(df, obj_id)[keys]
        if not target_df.empty:
            if len(keys) == 1:
                target = {keys[0]: target_df.iloc[0][keys[0]]}  # 단일 값 반환 as dict
            else:
                target = target_df.iloc[0].to_dict()  # 여러 개 값 dict 반환
            return target
        else:
            return {}
    
    elif eval['function_name'] in ["is_next_to"]:
        # TODO: entity_a, entity_b는 receptacle이 있는 경우도 있음. 따라서, 이를 해결하기 위한 로직 필요
        handle_a = eval['args']['entity_handles_a'][0]
        handle_b = eval['args']['entity_handles_b'][0]
        id_a = handle_a.split(":")[0][:-1]
        id_b = handle_b.split(":")[0][:-1]
        target_a_df = check_used_object(df, id_a)[keys]
        target_b_df = check_used_object(df, id_b)[keys]
        result = {}
        if not target_a_df.empty:
            if len(keys) == 1:
                result.update({keys[0]: target_a_df.iloc[0][keys[0]]})
            else:
                result.update(target_a_df.iloc[0].to_dict())
        if not target_b_df.empty:
            if len(keys) == 1:
                result.update({keys[0]: target_b_df.iloc[0][keys[0]]})
            else:
                result.update(target_b_df.iloc[0].to_dict())
        return result
    else:
        return {}
    
def check_used_handle(category: str, episode: list, df: pd.DataFrame) -> str:
    """
        category를 받아, df에서 조회해 해당하는 handle을 반환함
    """
    target_objects = episode['target_objects']
    for obj in target_objects:
        # target_a_df = df[df['category'] == category][keys]
        pass
        
def remove_duplicates(data):
    unique_data = list({tuple(item.items()): item for item in data}.values())
    return unique_data
        

def edit_path(data_path: str):
    return "data" + data_path.split('data')[-1]


class TaskPipeline:
    def __init__(self, 
                 data_path: str,
                #  tuned_instructions: List[Dict], 
                 dataset: Dict, 
                 distractor_sampling_fn: Callable, 
                 object_category_df: pd.DataFrame,
                 category_cluster_df: pd.DataFrame,
                 sampled_dataset: Dict,
                 object_semantics_dataset = Dict,
                 object_description_dataset = Dict,
                 user_pattern_dataset = Dict
                 ):  
        """
        :param dataset: 원본 instruction 파일 리스트 (List of Dict) /data/datasets/partnr_episodes/v0_0/train_2k.json
        :param tuned_instructions: 튜닝된 instruction 리스트 (List of Dict)
        :param distractor_sampling_fn: 데이터 샘플링 함수
        """
        self.data_path = edit_path(data_path)
        
        self.dataset = dataset # raw datasets
        
        self.sampled_dataset = sampled_dataset
        ## About both
        # self.tuned_instructions = tuned_instructions # augmented instructions
        
        ## About object semantics
        self.distractor_sampling_fn = distractor_sampling_fn # sampling function (아직 미정) # => (그냥 예시 속 데이터 가져오기)
        self.object_category_df = object_category_df # object의 category
        self.category_cluster_df = category_cluster_df # object의 caption과 cluster
        self.object_semantics_dataset = object_semantics_dataset
        self.object_description_dataset = object_description_dataset
        self.user_pattern_dataset = user_pattern_dataset
        
        self.final_dataset = {} # scene_id: [dataset_1, dataset_2, ...]

    def get_dataset_size(self):
        return len(self.dataset)
        
    def get_final_dataset_size(self):
        size = 0
        for k, datasets in self.final_dataset.items():
            for dataset in datasets:
                size += len(dataset)
        return size

    def pre_process(self):
        print(f"dataset size: {self.get_dataset_size()}")
        self.filter_by_scene()
        print(f"After filter_by_scene() -> dataset size: {self.get_dataset_size()}")
        self.preexclude_multi_eval()
        print(f"After preexclude_multi_eval() -> dataset size: {self.get_dataset_size()}")
        self.preexclude_multi_sampled()
        print(f"After preexclude_multi_sampled() -> dataset size: {self.get_dataset_size()}")
        self.preexclude_non_clusterd()
        print(f"After preexclude_non_clustered() -> dataset size: {self.get_dataset_size()}")
        self.prechange_handle()
        print(f"After prechange_handle() -> dataset size: {self.get_dataset_size()}")
        self.scenes = self.group_by_scene() # datas grouped by scenes
    
    def filter_by_scene(self):
        """
        sampled_dataset에는 없는 episode가 존재함. 따라서, 그걸 미리 filtering 하는 것이 목적
        """
        sampled_episode_ids = list(self.sampled_dataset.keys())
        sampled_episode_ids_set = set(sampled_episode_ids)  # 리스트를 set으로 변환
        self.dataset = [data for data in self.dataset if data['episode_id'] in sampled_episode_ids_set]

    def preexclude_multi_eval(self):
        dataset = []
        for data in self.dataset:
            evals = data['evaluation_propositions']
            if any(len(eval['args'].get('object_handles', [])) >= 2 for eval in evals):
                continue  
            dataset.append(data)
        self.dataset = dataset
        
    def preexclude_multi_sampled(self):
        dataset = []
        for data in self.dataset:
            initial_states = data['info']['initial_state']
            if any(not state.get('name') and state.get('number') is not None and int(state.get('number')) > 1 
                   for state in initial_states):
                continue  
            dataset.append(data)  
        self.dataset = dataset
        
    def prechange_handle(self):
        """
        caption이 없는 handle을 eval에 사용한다면 미리 바꿔주는 것이 목적
        """
        dataset = []
        for i, episode in enumerate(self.dataset):
            episode_id = episode['episode_id'] #########################
             
            evals = episode['evaluation_propositions']
            modified_evals = []  # Store modified evaluations separately

            obj_info_all_list = []
            obj_info_caption_list = []

            # First, collect all obj_info
            for eval in evals:
                obj_info_all = check_used_object_category_from_eval(eval, self.object_category_df, ['id', 'clean_category'])
                obj_info_caption = check_used_object_category_from_eval(eval, self.category_cluster_df, ['id', 'caption', 'source', 'category', 'cluster'])
                if obj_info_all:
                    obj_info_all_list.append(obj_info_all)
                if obj_info_caption:
                    obj_info_caption_list.append(obj_info_caption)
            
            obj_info_all_list = remove_duplicates(obj_info_all_list)
            obj_info_caption_list = remove_duplicates(obj_info_caption_list)
            if episode_id == "1385":
                print(obj_info_all_list)
                print(obj_info_caption_list)

            # Handle cases where obj_info_caption_list is empty
            if not obj_info_caption_list:
                diff_objs = obj_info_all_list  # If no caption data exists, all obj_info_all are considered different
            else:
                diff_objs = [obj_info_all for obj_info_all in obj_info_all_list 
                                if obj_info_all['id'] not in {obj_info_caption['id'] for obj_info_caption in obj_info_caption_list}]
            
            if episode_id == "1385":
                print(diff_objs)

            for obj_info_all in diff_objs:
                if sampled_obj := self.sample_by_category(obj_info_all, 'clean_category'):
                    # obj_info_all 부분을 sampled_obj로 변경해줘야 함.
                    old_id = obj_info_all['id']
                    new_id = sampled_obj['id']

                    # rigid_objs 수정
                    for obj in episode['rigid_objs']:
                        if old_id in obj[0]:
                            obj[0] = obj[0].replace(old_id, new_id)

                    # name_to_receptacle 수정
                    updated_dict = {}
                    for k, v in episode['name_to_receptacle'].items():
                        new_k = k.replace(old_id, new_id) if old_id in k else k
                        updated_dict[new_k] = v  # 새로운 key로 값 저장
                    episode['name_to_receptacle'] = updated_dict

                    # evaluation_propositions 수정
                    for eval_item in evals:
                        modified_eval = eval_item.copy()
                        if eval_item['function_name'] in ['is_on_top', 'is_in_room', 'is_inside']:
                            if old_id in eval_item['args']['object_handles'][0]:
                                modified_eval['args']['object_handles'][0] = eval_item['args']['object_handles'][0].replace(old_id, new_id)
                        elif eval_item['function_name'] == "is_next_to":
                            if old_id in eval_item['args']['entity_handles_a']:
                                modified_eval['args']['entity_handles_a'][0] = eval_item['args']['entity_handles_a'][0].replace(old_id, new_id)
                            if old_id in eval_item['args']['entity_handles_b']:
                                modified_eval['args']['entity_handles_b'][0] = eval_item['args']['entity_handles_b'][0].replace(old_id, new_id)
                        modified_evals.append(modified_eval)  # Store modified eval
                    
            if diff_objs:
                episode['evaluation_propositions'] = modified_evals  # Update evaluations after iteration
                
            dataset.append(episode)

        self.dataset = dataset
        
    def preexclude_non_clusterd(self):
        dataset = []
        for i, episode in enumerate(self.dataset):
            used_object_categories = self.extract_categories_from_episode(episode)
            if not self.check_objects_can_sampled(used_object_categories):
                continue  # 샘플링할 수 없는 경우 건너뛰기
            else:
                dataset.append(episode)
        
        self.dataset = dataset

    def check_objects_can_sampled(self, used_object_categories):
        for category in used_object_categories:
            if category not in self.category_cluster_df['category'].values:
                return False    
        return True

    def group_by_scene(self) -> Dict[str, List[Dict]]:
        """
        Scene ID 기준으로 에피소드 그룹화
        :return: scene_id를 key로 하는 episode dictionary ex. 203817140: [{}, {}]
        """
        scene_dict = defaultdict(list)
        for episode in self.dataset:
            scene_id = episode.get("scene_id", "unknown_scene")
            episode['used'] = False # 사용되었는지?
            scene_dict[scene_id].append(episode)
        return scene_dict

    def create_dataset(self):
        for _, (scene_id, episodes) in enumerate(tqdm(self.scenes.items())):
            self.final_dataset[scene_id] = self.create_dataset_per_scene(scene_id)
    
    def create_dataset_per_scene(self, scene_id: str):
        datasets = self.create_dataset_for_object(scene_id)
        datasets = self.create_dataset_for_user_pattern(scene_id, datasets)
        
        return datasets
    
    def check_objects_can_sampled(self, used_object_categories):
        for category in used_object_categories:
            if category not in self.category_cluster_df['category'].values:
                return False    
        return True

    
    def create_dataset_for_object(self, scene_id: str) -> List[Dict]:
        """
        Sampling per scene of original datasets.
        Each dataset contains 5 episodes.
        The process stops if the number of collected episodes exceeds half of the available episodes.
        The remaining half is used for the user pattern.
        """
        datasets = []
        episodes = self.scenes[scene_id]
        max_attempts = len(episodes) / 5  # 최대 시도 횟수
        attempt_count = 0

        while len(list(itertools.chain(*datasets))) < len(episodes) / 2 - 5:
            if attempt_count >= max_attempts:
                print(f"Max attempts reached. Stopping early for scene {scene_id}.")
                break  # Fail-safe: 일정 횟수 초과하면 종료

            sampled_episodes = []
            indices_to_update = []  # 'used' 상태를 업데이트할 인덱스 저장
            total_used_object_categories = []

            episode_indices = list(range(len(self.scenes[scene_id])))            
            random.seed(42)
            random.shuffle(episode_indices)

            for i in episode_indices:
                if self.scenes[scene_id][i]['used']:
                    continue  # 이미 사용된 에피소드는 스킵

                used_object_categories = self.extract_categories_from_episode(self.scenes[scene_id][i])
                sampled_episodes.append(self.scenes[scene_id][i])
                indices_to_update.append(i)
                total_used_object_categories.extend(used_object_categories)

                if len(sampled_episodes) == 5:  # 5개가 모이면 종료
                    break

            # 5개가 모인 경우에만 datasets 업데이트
            if len(sampled_episodes) == 5:
                datasets.append(sampled_episodes)
                for idx in indices_to_update:
                    self.scenes[scene_id][idx]['used'] = True  # 사용된 에피소드 표시
            else:
                print(f"Skipping dataset update: Not enough valid episodes available.")

            attempt_count += 1  # 시도 횟수 증가

        return datasets
        
    def create_dataset_for_user_pattern(self, scene_id: str, datasets: List):
        """
        Create dataset for user patterns. Ensures that only complete episode groups (size = 5) are added.
        If an incomplete dataset is created, it is removed from the list.
        """
        valid_datasets = []  # 최종적으로 유지할 datasets 리스트

        for dataset in datasets:
            episode_cand = []
            indices_to_update = []  # 'used' 상태를 변경할 인덱스 저장

            for j, episode in enumerate(self.scenes[scene_id]):
                if not self.scenes[scene_id][j]['used']:  # 아직 사용되지 않은 에피소드만 선택
                    episode_cand.append(episode)
                    indices_to_update.append(j)

                if len(episode_cand) == 5:  # 5개가 모이면 중단
                    break

            # 5개를 모았을 때만 업데이트
            if len(episode_cand) == 5:
                dataset.extend(episode_cand)
                for idx in indices_to_update:
                    self.scenes[scene_id][idx]['used'] = True  # 사용된 에피소드 표시
                valid_datasets.append(dataset)  # 유효한 데이터셋만 추가
            else:
                print(f"Skipping dataset: Not enough unused episodes available.")

        return valid_datasets  # 유효한 데이터셋만 반환

    def mid_process(self):
        # Step 1: extract_original_info 적용
        updated_final_dataset = {}
        for scene_id, datasets in self.final_dataset.items():
            updated_datasets = [extract_original_info(dataset, self.data_path) for dataset in datasets]
            updated_final_dataset[scene_id] = updated_datasets
        self.final_dataset = updated_final_dataset

        # Step 2: renew_episode_info 적용
        global_idx = 0
        updated_final_dataset = {}
        for scene_id, datasets in self.final_dataset.items():
            updated_datasets = []
            for dataset in datasets:
                updated_dataset = renew_episode_info(dataset, global_idx)
                global_idx += len(dataset)  # 사용된 episode 개수만큼 global index 증가
                updated_datasets.append(updated_dataset)
            updated_final_dataset[scene_id] = updated_datasets
        self.final_dataset = updated_final_dataset

        # Step 3: add_metadata 적용
        updated_final_dataset = {}
        for scene_id, datasets in self.final_dataset.items():
            updated_datasets = [add_metadata(dataset) for dataset in datasets]
            updated_final_dataset[scene_id] = updated_datasets
        self.final_dataset = updated_final_dataset

        # Step 4: add_stage_2 적용
        updated_final_dataset = {}
        for scene_id, datasets in self.final_dataset.items():
            updated_datasets = [add_stage_2(dataset) for dataset in datasets]
            updated_final_dataset[scene_id] = updated_datasets
        self.final_dataset = updated_final_dataset

    
    def extract_categories_from_episode(self, episode):
        evals = episode['evaluation_propositions']
        used_categories = []
            
        for eval in evals:
            used_object = check_used_object_category_from_eval(eval, self.object_category_df, ['clean_category'])
            if (clean_category := used_object.get('clean_category')) is not None:
                used_categories.append(clean_category)
        
        return used_categories
        

    def sample_distractors(self):
        for scene_id, datasets in self.final_dataset.items():
            for i, dataset in enumerate(datasets):
                self.final_dataset[scene_id][i] = self.get_target_objects_per_dataset(dataset)
                self.final_dataset[scene_id][i] = self.sample_distractors_per_dataset(dataset)
    
    def sample_by_category(self, obj_info, key):
        category = obj_info[key]
        filtered_df = self.category_cluster_df[self.category_cluster_df["category"] == category]
        if not filtered_df.empty:
            random_row = filtered_df.sample(n=1)  # 랜덤으로 하나 선택
            new_entry = random_row.to_dict(orient="records")[0]
            return new_entry
        else:
            return False
    
    def get_target_objects_per_dataset(self, dataset):
        """
        Dataset에서 각 episode의 target_objects를 추출하는 함수
        """
        for episode in dataset:
            evals = episode['evaluation_propositions']
            object_infos = []

            for eval in evals:
                obj_info_caption = check_used_object_category_from_eval(eval, self.category_cluster_df, ['id', 'caption', 'source', 'category', 'cluster'])
                if obj_info_caption:
                    object_infos.append(obj_info_caption)

            # Counter를 활용하여 중복 제거 (한 번만 등장한 것만 선택)
            unique_object_infos = list({tuple(obj.items()): obj for obj in object_infos}.values())

            # Target objects를 episode에 추가
            episode['metadata']['target_objects'] = unique_object_infos

        return dataset
    
    
    def sample_distractors_per_dataset(self, dataset):
        for i, episode in enumerate(dataset):
            if episode['metadata']['episode_type'] != "object_semantics": # user_pattern memorization only
                continue

            unique_object_infos = episode['metadata']['target_objects']
            distractors = []
            
            for obj in unique_object_infos:
                category = obj["category"]
                cluster = obj["cluster"]

                # 같은 category, 다른 cluster에서 샘플링
                filtered_df = self.category_cluster_df[
                    (self.category_cluster_df['category'] == category) &
                    (self.category_cluster_df['cluster'] != cluster)
                ]

                # 랜덤 샘플링 (해당 데이터가 있을 경우에만)
                if not filtered_df.empty:
                    random_row = filtered_df.sample(n=1).iloc[0]  # 랜덤 샘플링
                    random_obj_dict = random_row.to_dict()
                    random_obj_dict.update({'pair_id': obj['id']})
                    distractors.append(random_obj_dict)  # 결과 저장

            # Distractor 정보를 원본 episode에 추가
            episode['metadata']['distractors'] = distractors

        return dataset  # 수정된 dataset 반환
    
    def check_used_category_handle_from_episode(self, episode: dict, category: str):
        rigid_objs = episode['rigid_objs']
        targets = []
        for handle_config, trans in rigid_objs:
            handle = handle_config.split('.')[0]
            if check_used_object_category(df=self.object_category_df, obj_id=handle, category=category):
                targets.append(handle)

                try:
                    if episode['original_data_info']['episode_id'] == "1385":
                        print(handle)
                except Exception as e:
                    continue
                
        return targets
            
    def update_episodes(self, episodes: list): # final_dataset의 episodes
        for i, episode in enumerate(episodes):
            episode_id = episode['original_data_info']['episode_id']
                        
            if episode['metadata']['episode_type'] == 'object_semantics':
                object_semantics_data = self.object_semantics_dataset.get(episode_id, {})  # 기본값을 빈 dict로 설정
                
                # components for object_semantics
                object_semantic = object_semantics_data.get('stage1', "")
                object_semantic_stage2 = object_semantics_data.get('stage2', "")
                semantic_type = object_semantics_data.get('type', None)
                
                object_description = self.object_description_dataset.get(episode_id).get('description')
                
                if episode['metadata']['stage'] == "1":
                    tuned_instruction = episode['original_data_info']['instruction'] + object_description + object_semantic 
                    
                elif episode['metadata']['stage'] == "2":
                    tuned_instruction = object_semantic_stage2
                else:
                    raise Exception(f"episode_{episode_id}: episode['metadata']['stage'] errors")
                
                episode['metadata']['subtype'] = semantic_type
                episode['instruction'] = tuned_instruction
                
                # episode_id로 sample할 것 가져오기 (used_objs) V
                used_object = object_semantics_data.get('used_object', [])
                sampled_episode = self.sampled_dataset[episode_id] # 여기서 handle position 가져와 업데이트 할 것임
                
                # used_objs 속 category 대상으로 T / S 양쪽에서 handle 정보 임시 저장
                for obj_category in used_object:
                    if episode_id == "1385":
                        print(obj_category)
                        
                    targets = self.check_used_category_handle_from_episode(episode, obj_category)
                    sampled = self.check_used_category_handle_from_episode(sampled_episode, obj_category)
                    
                    # rigid_objs: Refactored transformation assignment
                    sampled_trans = []
                    for objs, trans in sampled_episode['rigid_objs']:
                        if objs.split('.')[0] in sampled:
                            sampled_trans.append(trans)

                    # Ensure there are enough transformations for both targets and distractors.
                    if len(sampled_trans) < 2 * len(targets):
                        raise ValueError("Not enough transformations for both targets and distractors.")

                    # Assign transformations: odd-indexed -> targets, even-indexed -> distractors
                    target_trans = [trans for i, trans in enumerate(sampled_trans) if i % 2 == 1]  # Odd-indexed
                    distractor_trans = [trans for i, trans in enumerate(sampled_trans) if i % 2 == 0]  # Even-indexed

                    # Assign target transformations
                    for i, target in enumerate(targets):
                        if i < len(target_trans):
                            for idx, (objs, origin_trans) in enumerate(episode['rigid_objs']):
                                if objs.split('.')[0] == target:
                                    episode['rigid_objs'][idx] = [objs, target_trans[i]]

                    # Assign distractor transformations
                    distractors = [
                        distractor for distractor in episode['metadata']['distractors']
                        if check_used_object_category(df=self.object_category_df, obj_id=distractor['id'], category=obj_category)
                    ]

                    for i, distractor in enumerate(distractors):
                        if i < len(distractor_trans):
                            episode['rigid_objs'].append([distractor['id'], distractor_trans[i]])
                        
                        
                    # name_to_receptacle
                    new_entries = {} 

                    for i, distractor in enumerate(distractors):
                        for k, v in episode['name_to_receptacle'].items(): 
                            if distractor.get('pair_id') in k:  
                                new_entries[distractor['id'] + "_:0000"] = v 

                    episode['name_to_receptacle'].update(new_entries)
                #   임시 정보 활용 본격 바꾸기 -> 단 조심해야 할 것은, T의 handle명은 유지하되, 좌표만 S에서 가져와야 함.
                #   따라서, S를 둘로 나눔. a) sample된 것 (info>extra_info>obj_info) b) 원래 것 (거기 없지만 카테고리 같은 것)
                
            
            elif episode['metadata']['episode_type'] == 'user_pattern':
                # components for user_pattern
                user_pattern_data = self.user_pattern_dataset.get(episode_id, {})
                user_pattern = user_pattern_data.get('stage1', "")
                user_pattern_stage2 = user_pattern_data.get('stage2', "")
                semantic_type = user_pattern_data.get('type', None)
                if episode['metadata']['stage'] == '1':
                    tuned_instruction = episode['original_data_info']['instruction'] + user_pattern
                elif episode['metadata']['stage'] == '2':
                    tuned_instruction = user_pattern_stage2
                else:
                    raise Exception(f"episode_{episode_id}: episode['metadata']['stage'] errors")
            
                episode['metadata']['subtype'] = semantic_type
                episode['instruction'] = tuned_instruction
            
            else:
                raise Exception(f"episode_{episode_id}: episode['metadata']['type'] errors")

                # let final_dataset (T) / sampled_dataset (S) V
                # T (original_data_info>episode_id)와 S (info>extra_info>episode_id)로 매칭 V
                # instruction 불러오기 V
                # tuned instruction 불러오기 (instruction: str = (inst+description+semantic), used_objs: list ([cup, tray])) V
            
                
    
    def change_handle(self, obj_category: str, episode: dict, sampled_episode: dict):
        # obj_category를 통해 V로 부터 가져오고 T에 반영해야 함
        #TODO: sampled_episode로부터 obj_category에 맞는 값 가져오기
        handle = check_used_handle(category=obj_category, episode=episode, df=self.category_cluster_df) # category_cluster_df 넣은 이유는 여기 밖의 것은 이미 필터링 해놔서, 여기서 안되면 어차피 터짐
        print(handle)
        #TODO: episode에 해당 값들 반영하기
        
        #TODO: episode에 distractor 변경하기
        
        pass


    def update_datasets(self):
        # self.final_dataset = {scene_id_1 = [[dataset_1], [dataset_2], ...]}
        for scene_id, datasets in self.final_dataset.items():
            for dataset in datasets:
                self.update_episodes(dataset)

    def change_instruction(self, episodes: list):
        for i, episode in enumerate(episodes):
            pass

    def update_instructions(self):
        for scene_id, datasets in self.final_dataset.items():
            pass
            
    def post_process(self):
        """
        1. 'used' 키 제거
        2. 최종 habitat dataset 형태 변환
        """
        for datasets in self.final_dataset.values():  # scene_id를 사용할 필요 없음
            for dataset in datasets:
                for episode in dataset:
                    episode.pop('used', None)  # 키가 없을 경우에도 안전하게 제거

        merged_dataset = [dataset for datasets in self.final_dataset.values() for dataset in datasets]
        self.final_dataset = merged_dataset
        

    def run(self) -> List[Dict]:
        """
        run the pipeline
        :return: final dataset
        """

        print("1. Filter not existing scenes, group by scene and change non-captioned handles...")
        self.pre_process()
        
        print("2. Sampling episodes per scene") # 2-1. Object semantics, 2-2. User pattern
        self.create_dataset() # self.final_dataset
        
        print("3. add metadata, original_info, process...")
        self.mid_process() # self.final_dataset

        print("4. sample distractors") # target, distractor sampling 완료
        self.sample_distractors()
        
        print("5. update instruction") # LLM call해서 description 추가하기 + 미리 만들어 둔 semantics 추가하기
        # self.update_instructions()
        
        print("6. update sample informations") # object sampling function 혹은 sampled dataset 활용
        self.update_datasets()
        # 1) object_sampling_function -> 
        # 2) sampled_dataset -> 해당하는 category 가져오기 => 그 category의 rigid_objs list만 가져오기 
        # => target과 distractor를 각각 대입
        # => name_to_receptacle, evaluation_propositions엔 target 여전히 사용
        print("7. postprocess")
        self.post_process()
        
        
        
        # df_tuple = self.preprocess_instruction_set()
        
        # print("🔹 Tuned Instruction과 Original File을 결합하여 Dataset 생성 중...")
        # self.create_dataset(scenes, df_tuple)
        # print(f" 총 {len(self.final_dataset)}개의 데이터 항목 생성 완료")
        # print("final", self.get_final_dataset_size())

        return self.final_dataset


def load_json(file_path: str) -> List[Dict]:
    with open(file_path, "r") as file:
        dataset = json.load(file)
    return dataset

def load_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

def df_to_dict(df: pd.DataFrame, key: str) -> Dict:
    dict_file = df.groupby(key).apply(lambda x: x.drop(columns=[key]).to_dict(orient='records')).to_dict()
    return dict_file

def test(datasets: dict):
    # 1. 네 번 사용된 에피소드 있으면 안됨
    episode_counts = Counter()

    for dataset in datasets:
        for data in dataset:
            episode_counts[data['episode_id']] += 1  # 각 episode_id의 등장 횟수를 카운트

    # 4번 이상 등장한 에피소드 찾기
    for episode_id, count in episode_counts.items():
        if count >= 4:
            print(f"⚠ Warning: Episode {episode_id} is used {count} times!")

    # 2. object_handles > 2인 것의 instruction 확인해보자.
    for dataset in datasets:
        for i, data in enumerate(dataset):
            if i <= 4:
                for eval in data['evaluation_propositions']:
                    if len(eval['args'].get('object_handles', [])) >= 2:
                        print(data['instruction'])
                    
    


if __name__ == "__main__":
    # raw dataset
    split = "train_2k"
    
    dataset_path = f"/HabitatLLM/data/datasets/partnr_episodes/v0_0/{split}.json"
    dataset: list[Dict] = load_json(dataset_path)['episodes']
    
    category_cluster_df: pd.DataFrame = load_csv("/HabitatLLM/data/hssd-hab/metadata/category_cluster.csv")

    # object categories
    obj_category_path: str = "/HabitatLLM/data/hssd-hab/metadata/object_categories_filtered.csv"
    obj_category_df: pd.DataFrame = pd.read_csv(obj_category_path)
    
    sampled_path: str = f"/HabitatLLM/workspace/dataset/{split}/merged_episodes.json" # 앞으로 이 형식으로 통일
    sampled_dataset = load_json(sampled_path)
    
    object_semantics_path: str = f"/HabitatLLM/workspace/vllm_inference/res/{split}/object_semantics_tuned.json"
    object_semantics_dataset = load_json(object_semantics_path)
    
    object_description_path: str = f"/HabitatLLM/workspace/vllm_inference/res/{split}/object_description_tuned.json"
    object_description_dataset = load_json(object_description_path)
    
    user_pattern_path: str = f"/HabitatLLM/workspace/vllm_inference/res/{split}/user_pattern_tuned.json"
    user_pattern_dataset = load_json(user_pattern_path)
    
    # user_pattern_path: str = f"/HabitatLLM/workspace/vllm_inference/res/user_pattern/{split}/seed_try.json"
    # user_pattern_dataset = load_json(user_pattern_path)
    
    # 더미 샘플링 함수 정의 (실제 구현 시 더 복잡한 로직 필요)
    def dummy_distractor_sampling(data):
        data["augmented"] = True
        return data

    pipeline = TaskPipeline(data_path = dataset_path,
                            dataset = dataset, 
                            distractor_sampling_fn = dummy_distractor_sampling,
                            object_category_df = obj_category_df,
                            category_cluster_df = category_cluster_df,
                            sampled_dataset = sampled_dataset,
                            object_semantics_dataset = object_semantics_dataset,
                            object_description_dataset = object_description_dataset,
                            user_pattern_dataset = user_pattern_dataset)
    
    final_dataset = pipeline.run()

    test(final_dataset)
    
    output_path = f"/HabitatLLM/workspace/preprocess/dataset/final_dataset_{split}_test.json"
    final_dataset = output_as_dataset(final_dataset, output_path)

