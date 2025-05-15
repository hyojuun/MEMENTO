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
                 object_category_df: pd.DataFrame,
                 category_cluster_df: pd.DataFrame,
                 sampled_dataset: Dict,
                 ):  
        """
        dataset 생성 위한 pre filtering 작업을 수행하는 pipeline임
        """
        self.data_path = edit_path(data_path)
        
        self.dataset = dataset # raw datasets
        
        self.sampled_dataset = sampled_dataset
        ## About both
        # self.tuned_instructions = tuned_instructions # augmented instructions
        
        ## About object semantics
        self.object_category_df = object_category_df # object의 category
        self.category_cluster_df = category_cluster_df # object의 caption과 cluster
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
        print("Changed non captioned handles")
        self.add_target_objects()
        print("Add target objects in the dataset")
        self.preexclude_non_rearrange()
        print(f"After preexclude_non_rearrange() -> dataset size: {self.get_dataset_size()}")
        self.preexclude_heterogeneous()
        print(f"After preexclude_heterogeneous() -> dataset size: {self.get_dataset_size()}")
    
    def filter_by_scene(self):
        """
        sampled_dataset에는 없는 episode가 존재함. 따라서, 그걸 미리 filtering 하는 것이 목적
        """
        sampled_episode_ids = list(self.sampled_dataset.keys())
        sampled_episode_ids_set = set(sampled_episode_ids)  # 리스트를 set으로 변환
        self.dataset = [data for data in self.dataset if data['episode_id'] in sampled_episode_ids_set]

    def preexclude_multi_eval(self):
        """
        같은 category 여러 개 sampling 해두고 그 중 하나 가져오기 이런 task 배제 1
        """
        dataset = []
        for data in self.dataset:
            evals = data['evaluation_propositions']
            if any(len(eval['args'].get('object_handles', [])) >= 2 for eval in evals):
                continue  
            dataset.append(data)
        self.dataset = dataset
        
    def preexclude_multi_sampled(self):
        """
        같은 category 여러 개 sampling 해두고 그 중 하나 가져오기 이런 task 배제 2
        """
        dataset = []
        for data in self.dataset:
            initial_states = data['info']['initial_state']
            if any(not state.get('name') and state.get('number') is not None and int(state.get('number')) > 1 
                   for state in initial_states):
                continue  
            dataset.append(data)  
        self.dataset = dataset
    
    def sample_by_category(self, obj_info, key: str):
        """
        obj_info의 category(or clean_category)를 통해 하나 random sampling 하는 함수
        """
        category = obj_info[key]
        filtered_df = self.category_cluster_df[self.category_cluster_df["category"] == category]
        if not filtered_df.empty:
            random_row = filtered_df.sample(n=1)  # 랜덤으로 하나 선택
            new_entry = random_row.to_dict(orient="records")[0]
            return new_entry
        else:
            return False
    
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

            # Handle cases where obj_info_caption_list is empty
            if not obj_info_caption_list:
                diff_objs = obj_info_all_list  # If no caption data exists, all obj_info_all are considered different
            else:
                diff_objs = [obj_info_all for obj_info_all in obj_info_all_list 
                                if obj_info_all['id'] not in {obj_info_caption['id'] for obj_info_caption in obj_info_caption_list}]
            
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
    
    def extract_categories_from_episode(self, episode):
        evals = episode['evaluation_propositions']
        used_categories = []
            
        for eval in evals:
            used_object = check_used_object_category_from_eval(eval, self.object_category_df, ['clean_category'])
            if (clean_category := used_object.get('clean_category')) is not None:
                used_categories.append(clean_category)
        
        return used_categories
    
    def preexclude_non_clusterd(self):
        """
        category cluster 없는 게 포함된 것들 skip 용도
        """
        dataset = []
        for i, episode in enumerate(self.dataset):
            used_object_categories = self.extract_categories_from_episode(episode)
            if not self.check_objects_can_sampled(used_object_categories):
                continue  # 샘플링할 수 없는 경우 건너뛰기
            else:
                dataset.append(episode)
        
        self.dataset = dataset
        
    def preinclude_metadata(self):
        """
        user_pattern 등에 쓰일 metadata를 미리 include해, dataset create 시에 활용하기 위함
        """
        dataset = []
        for i, episode in enumerate(self.dataset):
            user_pattern = self.user_pattern_dataset[episode['episode_id']]['type']
            episode['user_pattern'] = user_pattern
            dataset.append(episode)
        
        self.dataset = dataset

    def check_objects_can_sampled(self, used_object_categories):
        for category in used_object_categories:
            if category not in self.category_cluster_df['category'].values:
                return False    
        return True

    def preexclude_non_rearrange(self):
        dataset = []
        for i, episode in enumerate(self.dataset):
            if len(episode['target_objects']) == 0:
                continue
            else:
                dataset.append(episode)
        
        self.dataset = dataset
        
    def preexclude_heterogeneous(self):
        dataset = []
        for i, episode in enumerate(self.dataset):
            evals = episode['evaluation_propositions']
            skip = False
            for eval in evals:
                if eval['function_name'] in ['is_filled', 'is_clean', 'is_dirty', 'is_empty', 'is_powered_on', 'is_powered_off']:
                    skip = True
                    break
            if not skip:
                dataset.append(episode)
        
        self.dataset = dataset
        
    
    
    def add_target_objects(self):
        """
        Dataset에서 각 episode의 target_objects를 추출하는 함수
        """
        for episode in self.dataset:
            evals = episode['evaluation_propositions']
            object_infos = []

            for eval in evals:
                obj_info_caption = check_used_object_category_from_eval(eval, self.category_cluster_df, ['id', 'caption', 'source', 'category', 'cluster'])
                if obj_info_caption:
                    object_infos.append(obj_info_caption)

            # Counter를 활용하여 중복 제거 (한 번만 등장한 것만 선택)
            unique_object_infos = list({tuple(obj.items()): obj for obj in object_infos}.values())

            # Target objects를 episode에 추가
            episode['target_objects'] = unique_object_infos

        return dataset

    def run(self) -> List[Dict]:
        """
        run the pipeline
        """

        print("1. Filter not existing scenes, group by scene and change non-captioned handles...")
        self.pre_process()
        
        return self.dataset


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
    
    dataset_path = f"dataset/inputs/partnr_episodes/v0_0/{split}.json"
    dataset: list[Dict] = load_json(dataset_path)['episodes']
    
    category_cluster_df: pd.DataFrame = load_csv("dataset/inputs/metadata/category_cluster.csv")

    # object categories
    obj_category_path: str = "dataset/inputs/metadata/object_categories_filtered.csv"
    obj_category_df: pd.DataFrame = pd.read_csv(obj_category_path)
    
    sampled_path: str = f"dataset/inputs/{split}/merged_episodes_{split}.json"
    sampled_dataset = load_json(sampled_path)

    pipeline = TaskPipeline(data_path = dataset_path,
                            dataset = dataset, 
                            object_category_df = obj_category_df,
                            category_cluster_df = category_cluster_df,
                            sampled_dataset = sampled_dataset)
    
    final_dataset = pipeline.run()
    
    output_path = f"dataset/inputs/{split}/pre_dataset_{split}.json"
    final_dataset = output_as_dataset(final_dataset, output_path)

