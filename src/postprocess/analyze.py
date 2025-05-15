import argparse
import json
import pandas as pd
import os

def get_args():
    parser = argparse.ArgumentParser(description='Process some data files.')
    parser.add_argument('--original_path', type=str, required=True, help='Path to the original JSON file')
    parser.add_argument('--anal_path', type=str, required=True, help='Path to the analysis JSON file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output CSV file')
    parser.add_argument('--memory_path', type=str, required=True, help='Path to the memory CSV file')
    args = parser.parse_args()
    return args

def load_data(original_path, anal_path):
    with open(original_path, "r") as f:
        data = json.load(f)['episodes']

    with open(anal_path, "r") as f:
        anal_data = json.load(f)

    return data, anal_data

def prepare_anal_df(anal_data):
    anal_df = pd.DataFrame(anal_data)[['episode_id', 'metadata']]
    anal_df['episode_id'] = anal_df['episode_id'].astype(int)

    metadata_df = pd.json_normalize(anal_df['metadata'])
    anal_df = pd.concat([anal_df.drop(columns=['metadata']), metadata_df[['original_type', 'dependent_rearrange', 'multistep']]], axis=1)
    return anal_df

def get_anal(memory_df, output_df, anal_df):
    def compute_type_ratios(df):
        all_types = df['original_type'].explode()
        total = len(all_types)
        counter = all_types.value_counts(normalize=True)
        return counter

    merged_df = pd.merge(memory_df, anal_df, on='episode_id', how='inner')
    merged_df['episode_id'] = merged_df['episode_id'] + 2000
    merged_df = pd.merge(merged_df, output_df, on='episode_id', suffixes=('_memory', '_output'))

    comparison_counts = merged_df.groupby(['task_state_success_memory', 'task_state_success_output']).size()

    print("\nComparison Counts:")
    print(comparison_counts)

    case_00 = merged_df[(merged_df['task_state_success_memory'] == 0) & (merged_df['task_state_success_output'] == 0)]
    case_01 = merged_df[(merged_df['task_state_success_memory'] == 0) & (merged_df['task_state_success_output'] == 1)]
    case_10 = merged_df[(merged_df['task_state_success_memory'] == 1) & (merged_df['task_state_success_output'] == 0)]
    case_11 = merged_df[(merged_df['task_state_success_memory'] == 1) & (merged_df['task_state_success_output'] == 1)]

    print("\nCase (Memory=0, Output=0):")
    print(case_00[['episode_id', 'original_type', 'multistep']])
    print(len(case_00))

    print("\nCase (Memory=0, Output=1):")
    print(case_01[['episode_id', 'original_type', 'multistep']])
    print(len(case_01))

    print("\nCase (Memory=1, Output=0):")
    print(case_10[['episode_id', 'original_type', 'multistep']])
    print(len(case_10))

    print("\nCase (Memory=1, Output=1):")
    print(case_11[['episode_id', 'original_type', 'multistep']])
    print(len(case_11))
    case_11 = case_11.sort_values(by=['episode_id'])
    case_11.to_csv("allright.csv")

    print("Case (0, 0):")
    print(compute_type_ratios(case_00))
    print("\nCase (0, 1):")
    print(compute_type_ratios(case_01))
    print("\nCase (1, 0):")
    print(compute_type_ratios(case_10))
    print("\nCase (1, 1):")
    print(compute_type_ratios(case_11))

def get_matrix(memory_df, output_df):
    merged_df = pd.merge(memory_df, output_df, on='episode_id', suffixes=('_memory', '_output'))
    matrix = merged_df.groupby(['task_state_success_output', 'task_state_success_memory']).size().unstack(fill_value=0)
    matrix = matrix.reindex(index=[1, 0], columns=[1, 0], fill_value=0)
    matrix['Total'] = matrix.sum(axis=1)
    matrix.loc['Total'] = matrix.sum()

    print("Task State Success Comparison Matrix (Reordered):")
    print(matrix)

def get_ratio(memory_df, output_df, anal_df):
    merged_df = pd.merge(memory_df, anal_df, on='episode_id', how='inner')
    merged_df['episode_id'] = merged_df['episode_id'] + 2000
    merged_df = pd.merge(merged_df, output_df, on='episode_id', suffixes=('_memory', '_output'))

    merged_df['type_group'] = merged_df['original_type'].apply(lambda x: str(sorted(x)))

    grouped = merged_df.groupby(
        ['type_group', 'multistep', 'task_state_success_memory', 'task_state_success_output']
    ).size().reset_index(name='count')

    grouped['ratio'] = grouped.groupby(['type_group', 'multistep'])['count'].transform(lambda x: x / x.sum())

    pivot_count = grouped.pivot_table(
        index=['type_group', 'multistep'],
        columns=['task_state_success_memory', 'task_state_success_output'],
        values='count',
        fill_value=0
    )
    pivot_count.columns = [f"({int(m)},{int(o)})" for m, o in pivot_count.columns]
    pivot_count = pivot_count.reset_index()

    pivot_ratio = grouped.pivot_table(
        index=['type_group', 'multistep'],
        columns=['task_state_success_memory', 'task_state_success_output'],
        values='ratio',
        fill_value=0
    )
    pivot_ratio.columns = [f"({int(m)},{int(o)})" for m, o in pivot_ratio.columns]
    pivot_ratio = pivot_ratio.reset_index()

    print("🟢 [개수]")
    print(pivot_count)

    print("\n🔵 [비율]")
    print(pivot_ratio)

    return pivot_count, pivot_ratio

def main(args):
    _, anal_data = load_data(args.original_path, args.anal_path)
    anal_df = prepare_anal_df(anal_data)

    output_df = pd.read_csv(args.output_path)
    memory_df = pd.read_csv(args.memory_path)

    get_anal(memory_df, output_df, anal_df)
    get_ratio(memory_df, output_df, anal_df)
    get_matrix(memory_df, output_df)

if __name__ == "__main__":
    args = get_args()
    main(args)