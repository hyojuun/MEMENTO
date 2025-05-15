import json
from collections import defaultdict, Counter
from typing import List, Dict, Callable
import time

import random
import itertools
from tqdm import tqdm

import numpy as np
import pandas as pd

from utils import output_as_dataset, renew_episode_info, add_metadata, extract_original_info, add_stage_2


np.random.seed(int(time.time()))

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

class TaskPipeline:
    def __init__(self, 
                 data_path: str,
                 split: str,
                 dataset: Dict, 
                 object_category_df: pd.DataFrame,
                 category_cluster_df: pd.DataFrame,
                 sampled_dataset: Dict,
                 object_semantics_dataset = Dict,
                 object_description_dataset = Dict,
                 user_pattern_dataset = Dict
                 ):  
        """
        Task pipeline that processes the dataset grouped by scene that unified all episodes.
        """
        self.data_path = data_path
        self.split = split
        
        self.dataset = dataset # preprocessed dataset
        self.sampled_dataset = sampled_dataset # sampled dataset
        
        ## About object semantics
        self.object_category_df = object_category_df
        self.category_cluster_df = category_cluster_df
        self.object_semantics_dataset = object_semantics_dataset
        self.object_description_dataset = object_description_dataset
        
        ## About user patterns
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

    def preinclude_user_pattern_data(self):
        """
        dataset grouping 과정에서 사용하기 위해 user_pattern data를 할당
        self.dataset = List[Dict]
        """
        for episode in self.dataset:
            episode_id = episode['episode_id']
            episode['user_pattern_type'] = self.user_pattern_dataset[episode_id]['type']
        

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

    def pre_process(self):
        self.preinclude_user_pattern_data()
        self.scenes = self.group_by_scene() # datas grouped by scenes


    def check_objects_can_sampled(self, used_object_categories):
        for category in used_object_categories:
            if category not in self.category_cluster_df['category'].values:
                return False    
        return True


    def create_dataset(self):
        for _, (scene_id, episodes) in enumerate(tqdm(self.scenes.items())):
            self.final_dataset[scene_id] = self.create_dataset_per_scene(scene_id)
    
    def create_dataset_per_scene(self, scene_id: str):
        stage_1_dataset = self.create_dataset_for_user_pattern(scene_id)
        stage_2_dataset = self.create_dataset_for_object(scene_id)
        
        return [stage_1_dataset, stage_2_dataset]

    def create_dataset_for_user_pattern(self, scene_id: str):
        """
        Create dataset for user patterns. Ensures that only complete episode groups (size = 5) are added.
        If an incomplete dataset is created, it is removed from the list.
        """
        valid_datasets = []  # 최종적으로 유지할 datasets
        episode_indices = list(range(len(self.scenes[scene_id])))
        
        for i in episode_indices:
            if self.scenes[scene_id][i]['user_pattern_type'] == "none":
                continue # user_pattern 없으면 사용하지 않음
            else:
                valid_datasets.append(self.scenes[scene_id][i].copy())
                
        return valid_datasets 
    
    def create_dataset_for_object(self, scene_id: str) -> List[Dict]:
        """
        Sampling per scene of original datasets.
        Each dataset contains 5 episodes.
        The process stops if the number of collected episodes exceeds half of the available episodes.
        The remaining half is used for the user pattern.
        """
        valid_datasets = []  # 최종적으로 유지할 datasets
        episode_indices = list(range(len(self.scenes[scene_id])))
        
        for i in episode_indices:
            if self.scenes[scene_id][i]['target_objects'] != 0:  
                valid_datasets.append(self.scenes[scene_id][i].copy())
        
        return valid_datasets

    def mid_process(self):
        # Step 1: extract_original_info 적용
        updated_final_dataset = {}
        for scene_id, datasets in self.final_dataset.items():
            updated_datasets = [extract_original_info(dataset, self.data_path) for dataset in datasets]
            updated_final_dataset[scene_id] = updated_datasets
        self.final_dataset = updated_final_dataset

        # Step 2: renew_episode_info 적용
        global_idx = 934
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
            updated_datasets = [add_metadata(dataset, self.split, idx) for idx, dataset in enumerate(datasets)]
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
            episode['metadata']['target_objects'] = episode['target_objects']

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
                
        return targets
    
    def update_instruction_per_dataset(self, episodes: list):
        for i, episode in enumerate(episodes):
            episode_id = episode['original_data_info']['episode_id']
                        
            if episode['metadata']['episode_type'] == 'object_semantics':
                object_semantics_data = self.object_semantics_dataset.get(episode_id, {})  # 기본값을 빈 dict로 설정
                
                # components for object_semantics
                object_semantic = object_semantics_data.get('stage1', "")
                object_semantic_stage2 = object_semantics_data.get('stage2', "")
                semantic_type = object_semantics_data.get('type')
                
                object_description = self.object_description_dataset.get(episode_id).get('description')
                
                if episode['metadata']['stage'] == "1":
                    tuned_instruction = f"{episode['original_data_info']['instruction']} {object_description} {object_semantic}" 
                    
                elif episode['metadata']['stage'] == "2":
                    tuned_instruction = object_semantic_stage2
                else:
                    raise Exception(f"episode_{episode_id}: episode['metadata']['stage'] errors")
                
                episode['metadata']['subtype'] = semantic_type
                tuned_instruction = tuned_instruction.replace('\\"', '"').replace('"', ' ').replace('\n', ' ').strip()
                tuned_instruction = tuned_instruction.replace('\\"', '"').replace('"', ' ').replace('\n', ' ').strip()
                episode['instruction'] = tuned_instruction
            
            elif episode['metadata']['episode_type'] == 'user_pattern':
                # components for user_pattern
                user_pattern_data = self.user_pattern_dataset.get(episode_id, {})
                user_pattern = user_pattern_data.get('stage1', "")
                user_pattern_stage2 = user_pattern_data.get('stage2', "")
                semantic_type = user_pattern_data.get('type')
                if episode['metadata']['stage'] == '1':
                    tuned_instruction = f"{episode['original_data_info']['instruction']} {user_pattern}"
                elif episode['metadata']['stage'] == '2':
                    tuned_instruction = user_pattern_stage2
                else:
                    raise Exception(f"episode_{episode_id}: episode['metadata']['stage'] errors")
            
                episode['metadata']['subtype'] = semantic_type
                tuned_instruction = tuned_instruction.replace('\\"', '"').replace('"', ' ').replace('\n', ' ').strip()
                tuned_instruction = tuned_instruction.replace('\\"', '"').replace('"', ' ').replace('\n', ' ').strip()
                episode['instruction'] = tuned_instruction
            
            else:
                raise Exception(f"episode_{episode_id}: episode['metadata']['type'] errors")        
    
    def update_instructions(self):
        # self.final_dataset = {scene_id_1 = [[dataset_1], [dataset_2], ...]}
        for scene_id, datasets in self.final_dataset.items():
            for dataset in datasets:
                self.update_instruction_per_dataset(dataset)
    
    def apply_sampled_transformations(self, episodes: list): # final_dataset의 episodes
        for i, episode in enumerate(episodes):
            episode_id = episode['original_data_info']['episode_id']
                        
            if episode['metadata']['episode_type'] == 'object_semantics':
                object_semantics_data = self.object_semantics_dataset.get(episode_id, {})  # 기본값을 빈 dict로 설정
                
                # episode_id로 sample할 것 가져오기 (used_objs) V
                used_object = object_semantics_data.get('used_object', [])
                sampled_episode = self.sampled_dataset[episode_id] # 여기서 handle position 가져와 업데이트 할 것임
                total_targets = 0
                
                # used_objs 속 category 대상으로 T / S 양쪽에서 handle 정보 임시 저장
                for obj_category in used_object:
                    targets = self.check_used_category_handle_from_episode(episode, obj_category)
                    total_targets += len(targets)
                    sampled = self.check_used_category_handle_from_episode(sampled_episode, obj_category)
                    
                    # rigid_objs: Refactored transformation assignment
                    sampled_trans = []
                    for objs, trans in sampled_episode['rigid_objs']:
                        if objs.split('.')[0] in sampled:
                            sampled_trans.append(trans)

                    # Ensure there are enough transformations for both targets and distractors.
                    if len(sampled_trans) < 2 * len(targets):
                        print(sampled_trans)
                        print(targets)
                        print(episode['original_data_info'])
                        print(episode['metadata'])
                        
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
                
                episode['metadata']['num_targets'] = total_targets
            
            else:
                continue


    def apply_sampled_data_to_final_dataset(self):
        # self.final_dataset = {scene_id_1 = [[dataset_1], [dataset_2], ...]}
        for scene_id, datasets in self.final_dataset.items():
            for dataset in datasets:
                self.apply_sampled_transformations(dataset)

            
    def post_process(self):
        """
        1. 'used' 키 제거
        2. 'target_objects' 키 제거
        3. 'user_pattern_type' 키 제거
        4. 최종 habitat dataset 형태 변환
        """
        for datasets in self.final_dataset.values(): 
            for dataset in datasets:
                for episode in dataset:
                    episode.pop('used', None) 
                    episode.pop('target_objects', None)
                    episode.pop('user_pattern_type', None)
        # merged_dataset = [dataset for datasets in self.final_dataset.values() for dataset in datasets]
        # self.final_dataset = merged_dataset
        

    def run(self) -> List[Dict]:
        """
        run the pipeline
        :return: final dataset
        """

        print("1. Grouping datasets by scenes...")
        self.pre_process()
        
        print("2. Sampling episodes per scene") # 2-1. Object semantics, 2-2. User pattern
        self.create_dataset() # self.final_dataset
        
        print("3. add metadata, original_info, process...")
        self.mid_process() # self.final_dataset

        print("4. sample distractors") # target, distractor sampling 완료
        self.sample_distractors()
        
        print("5. update instruction") # LLM call해서 description 추가하기 + 미리 만들어 둔 semantics 추가하기
        self.update_instructions()
        
        print("6. update sample informations") # object sampling function 혹은 sampled dataset 활용
        self.apply_sampled_data_to_final_dataset()
        
        print("7. postprocess")
        self.post_process()
        
        print(f"total {len(self.final_dataset)} datasets")

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
    split = "val"
    
    ## dataset after prepipeline ##
    predataset_path = f"dataset/inputs/{split}/pre_dataset_{split}.json"
    dataset: list[Dict] = load_json(predataset_path)['episodes']
    
    dataset_path = f"data/datasets/partnr_episodes/v0_0/{split}.json"
    
    ## datasets for object semantics ##
    
    #### objects with (id, caption, source, category, cluster). only contains objects with caption.
    category_cluster_df: pd.DataFrame = load_csv("dataset/inputs/metadata/category_cluster.csv")

    #### objects with (id, clean_category). contains all objects.
    obj_category_path: str = "dataset/inputs/metadata/object_categories_filtered.csv"
    obj_category_df: pd.DataFrame = pd.read_csv(obj_category_path)
    
    #### dataset with sampled objects informations. (by SJ)
    sampled_path: str = f"dataset/inputs/{split}/sampled_info_{split}.json"
    sampled_dataset = load_json(sampled_path)
    
    #### dataset with object semantics augmented informations. (by GPT-4o) ex. My cup
    object_semantics_path: str = f"dataset/inputs/{split}/object_semantics_tuned_{split}.json"
    object_semantics_dataset = load_json(object_semantics_path)
    
    #### dataset with object description augmented informations. (by GPT-4o) ex. Red cup
    object_description_path: str = f"dataset/inputs/{split}/object_description_tuned_{split}.json"
    object_description_dataset = load_json(object_description_path)
    
    
    ## datasets for user patterns ##
    
    #### dataset with user pattern augmented informations. (by GPT-4o) ex. Movie night setting
    user_pattern_path: str = f"dataset/inputs/{split}/user_pattern_tuned_{split}.json"
    user_pattern_dataset = load_json(user_pattern_path)

    ## Create the pipeline instance
    pipeline = TaskPipeline(data_path = dataset_path,
                            split = split,
                            dataset = dataset, 
                            object_category_df = obj_category_df,
                            category_cluster_df = category_cluster_df,
                            sampled_dataset = sampled_dataset,
                            object_semantics_dataset = object_semantics_dataset,
                            object_description_dataset = object_description_dataset,
                            user_pattern_dataset = user_pattern_dataset)
    
    final_dataset = pipeline.run()

    ## Final test the dataset (#TODO: task pipeline 안으로 통합)
    # test(final_dataset)
    
    ## Final output
    output_path = f"dataset/outputs/post_dataset_v2_{split}.json"
    final_dataset = output_as_dataset(final_dataset, output_path)


# 만약 바꾼다면, 