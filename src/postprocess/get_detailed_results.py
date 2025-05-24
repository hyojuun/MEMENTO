import argparse
import os
import json
import pandas as pd

ORIGINAL_STAGE1_DATA_PATH = "data/datasets/PEAD/analysis/stage_1_data.json"
ORIGINAL_STAGE2_DATA_PATH = "data/datasets/PEAD/analysis/stage_2_data.json"
ORIGINAL_STAGE3_DATA_PATH = "data/datasets/PEAD/analysis/stage_3_data.json"

STAGE1_SUBCATEGORY_LIST = ["history", "group", "preference", "ownership"]
STAGE2_SUBCATEGORY_LIST = ["preference", "routine"]
# Stage3의 복합 타입 정의
STAGE3_TYPE_CATEGORIES = [
    ["object_semantics", "object_semantics"],
    ["object_semantics", "user_pattern"],
    ["user_pattern", "user_pattern"]
]
# Stage3의 각 타입별 하위 카테고리 매핑
STAGE3_SUBCATEGORY_DICT = {
    "object_semantics-object_semantics": ["history-history", "group-group", "preference-preference", 
                                        "ownership-ownership", "history-group", "history-preference", 
                                        "history-ownership", "group-preference", "group-ownership", 
                                        "preference-ownership"],
    "object_semantics-user_pattern": ["history-preference", "history-routine", "group-preference", 
                                    "group-routine", "preference-preference", "preference-routine", 
                                    "ownership-preference", "ownership-routine"],
    "user_pattern-user_pattern": ["preference-preference", "preference-routine", "routine-routine"]
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default="/MEMENTO/data/outputs/v0_val/gpt-4o_object-semantics-stage1_04-09_14-48/results/episode_log.csv")
    parser.add_argument("--type", type=str, choices=["object_semantics", "user_pattern", "all"], default="all", help="Type of results to print")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=1, help="Stage of results to print")
    parser.add_argument("--save_dir", type=str, default="/MEMENTO/results/gpt-4o_object-semantics-stage1_04-09_14-48")
    parser.add_argument("--do_integrate", action="store_true", help="Integrate results from different stages")
    parser.add_argument("--integration_path_list", type=str, nargs="+", help="Path of the results to integrate")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    return args

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)
    
def read_csv(path):
    return pd.read_csv(path)

def convert_df_to_list(df, org_stage_data):
    data = []
    for _, row in df.iterrows():
        tmp = row.to_dict()
        for d in org_stage_data:
            if int(d['episode_id']) == int(tmp['episode_id']):
                tmp['org_data'] = d
                break
        data.append(tmp)
    return data

def get_mean_results(episode_result_list):
    # get mean results
    mean_results = {}
    for result in episode_result_list:
        ## get result data
        for key, value in result.items():
            if key == "episode_id" or key == "instruction" or key == "org_data" or key == "error":
                continue
            if key not in mean_results:
                mean_results[key] = []
            mean_results[key].append(value)
    
    for key, value in mean_results.items():
        mean_results[key] = sum(value) / len(value)
    
    return mean_results

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
        
        
def filter_episode_result_df(episode_result_df):
    filter_episode_id_list = [1090, 1227,  965, 1086, 1015, 1182, 1125, 1066, 1247, 1139, 1172, 1170, 1153, 1101, 1152, 1252, 1157, 1156, 1136, 1267, 1158, 1121, 1112, 1197, 1208, 1194,  976, 1167, 1265,  993, 1192]
    filter_episode_id_list2 = [3090, 3227, 2965, 3086, 3015, 3182, 3125, 3066, 3247, 3139, 3172, 3170, 3153, 3101, 3152, 3252, 3157, 3156, 3136, 3267, 3158, 3121, 3112, 3197, 3208, 3194,  2976, 3167, 3265,  2993, 3192]
    
    # Exclude episodes with IDs in the filter lists
    return episode_result_df[~episode_result_df['episode_id'].isin(filter_episode_id_list + filter_episode_id_list2)]

def integrate_paths(path_list):
    # integrate csv files in the path_list
    total_csv = pd.DataFrame()
    for path in path_list:
        df = read_csv(path)
        total_csv = pd.concat([total_csv, df])
        
    # episode_id 중복 제거
    total_csv = total_csv.drop_duplicates(subset=['episode_id'])
    return total_csv

def main(args):
    
    if args.do_integrate:
        episode_result_df = integrate_paths(args.integration_path_list)
    else:
        episode_result_path = args.result_path
        episode_result_df = read_csv(episode_result_path)
    
    # filter episode result df by episode_id
    episode_result_df = filter_episode_result_df(episode_result_df)
    
    # print episode_result length
    print(f"Total number of episodes: {len(episode_result_df)}")
    
    # read original data
    if args.stage == 1:
        # read stage 1 data
        org_stage_1_data = read_json(ORIGINAL_STAGE1_DATA_PATH)
        episode_result_list = convert_df_to_list(episode_result_df, org_stage_1_data)
        
        # Stage 1 subcategory dictionary
        subcategory_dict = {
            "object_semantics": {key: {"total": [], "correct": [], "incorrect": []} for key in STAGE1_SUBCATEGORY_LIST},
            "user_pattern": {key: {"total": [], "correct": [], "incorrect": []} for key in STAGE2_SUBCATEGORY_LIST}
        }
    elif args.stage == 2:
        # read stage 2 data
        org_stage_2_data = read_json(ORIGINAL_STAGE2_DATA_PATH)
        episode_result_list = convert_df_to_list(episode_result_df, org_stage_2_data)
        
        # Stage 2 subcategory dictionary
        subcategory_dict = {
            "object_semantics": {key: {"total": [], "correct": [], "incorrect": []} for key in STAGE1_SUBCATEGORY_LIST},
            "user_pattern": {key: {"total": [], "correct": [], "incorrect": []} for key in STAGE2_SUBCATEGORY_LIST}
        }
    elif args.stage == 3:
        # read stage 3 data
        org_stage_3_data = read_json(ORIGINAL_STAGE3_DATA_PATH)
        episode_result_list = convert_df_to_list(episode_result_df, org_stage_3_data)
        
        # Stage 3 subcategory dictionary - 복합 타입 처리
        subcategory_dict = {}
        for type_combo in STAGE3_TYPE_CATEGORIES:
            type_key = f"{type_combo[0]}-{type_combo[1]}"
            subcategory_dict[type_key] = {
                subtype: {"total": [], "correct": [], "incorrect": []} 
                for subtype in STAGE3_SUBCATEGORY_DICT[type_key]
            }
    
    print(f"Total number of episodes: {len(episode_result_list)}")
    
    # get detailed results
    mean_results = get_mean_results(episode_result_list)
    
    # Print mean results
    print(f"The Total results of {args.type} of Stage {args.stage}:")
    for key, value in mean_results.items():
        print(f"{key}: {value}")
    
    # Group results by subtype
    for episode in episode_result_list:
        if args.stage < 3:
            # Stage 1,2 처리 (기존 코드)
            episode_type = episode['org_data']['metadata']['episode_type']
            subtype = episode['org_data']['metadata']['subtype']
            
            # 해당 타입과 하위타입이 subcategory_dict에 있는지 확인
            if episode_type in subcategory_dict and subtype in subcategory_dict[episode_type]:
                if episode['task_state_success']:
                    subcategory_dict[episode_type][subtype]["correct"].append(episode)
                else:
                    subcategory_dict[episode_type][subtype]["incorrect"].append(episode)
                subcategory_dict[episode_type][subtype]["total"].append(episode)
        else:
            # Stage 3 처리 (복합 타입)
            # Stage 3의 메타데이터 구조는 원본 데이터를 확인해야 함
            # 여기서는 가정한 구조로 작성했으니 실제 데이터 구조에 맞게 조정 필요
            types = sorted(episode['org_data']['metadata']['episode_type'])
            type_1 = types[0]
            type_2 = types[1]
            subtypes = sorted(episode['org_data']['metadata']['subtype'])
            subtype_1 = subtypes[0]
            subtype_2 = subtypes[1]
            
            # 복합 키 생성
            type_key = f"{type_1}-{type_2}"
            subtype_key = f"{subtype_1}-{subtype_2}"
            
            # 해당 타입과 하위타입이 subcategory_dict에 있는지 확인
            if type_key in subcategory_dict and subtype_key in subcategory_dict[type_key]:
                if episode['task_state_success']:
                    subcategory_dict[type_key][subtype_key]["correct"].append(episode)
                else:
                    subcategory_dict[type_key][subtype_key]["incorrect"].append(episode)
                subcategory_dict[type_key][subtype_key]["total"].append(episode)
    
    # 나머지 코드는 기존과 동일하게 유지
    # Calculate and save mean results for each subcategory
    subcategory_mean_results = []
    for episode_type, subtypes in subcategory_dict.items():
        for subtype, results in subtypes.items():
            if results["total"]:
                subcategory_mean = get_mean_results(results["total"])
                subcategory_mean['name'] = f"{episode_type}_{subtype}"
                subcategory_mean_results.append(subcategory_mean)
    
    # Calculate total results for each type
    for episode_type, subtypes in subcategory_dict.items():
        total_results = {"total": [], "correct": [], "incorrect": []}
        for results in subtypes.values():
            total_results["total"].extend(results["total"])
            total_results["correct"].extend(results["correct"])
            total_results["incorrect"].extend(results["incorrect"])
        
        if total_results["total"]:
            type_mean = get_mean_results(total_results["total"])
            type_mean['name'] = f"{episode_type}_total"
            subcategory_mean_results.append(type_mean)

    # Add total mean results to the subcategory mean results
    total_mean_results = get_mean_results(episode_result_list)
    total_mean_results['name'] = 'total_results'
    subcategory_mean_results.append(total_mean_results)
    
    # Convert to DataFrame and reorder columns
    subcategory_mean_results_df = pd.DataFrame(subcategory_mean_results)
    
    # Ensure the DataFrame has the specified columns
    columns_order = ['name', 'replanning_count_0', 'runtime', 'sim_step_count', 'task_percent_complete', 'task_state_success']
    subcategory_mean_results_df = subcategory_mean_results_df.reindex(columns=columns_order)
    
    subcategory_mean_results_csv_path = os.path.join(args.save_dir, f"results_stage_{args.stage}_{args.type}.csv")
    subcategory_mean_results_df.to_csv(subcategory_mean_results_csv_path, index=False)
    print(f"Subcategory mean results saved to {subcategory_mean_results_csv_path}")
    
    
    for key, value in subcategory_dict.items():
        file_data_dir = os.path.join(args.save_dir, f"{key}")
        
        # Check if the value is not empty
        if not any(results["total"] for results in value.values()):
            continue
        
        os.makedirs(file_data_dir, exist_ok=True)
        
        for subtype, results in value.items():
            save_json(results["total"], os.path.join(file_data_dir, f"{subtype}_total.json"))
            save_json(results["correct"], os.path.join(file_data_dir, f"{subtype}_correct.json"))
            save_json(results["incorrect"], os.path.join(file_data_dir, f"{subtype}_incorrect.json"))
                
        
    
    
    return
    
    
if __name__ == "__main__":
    args = get_args()
    main(args)
