import json
import os
import argparse
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/instructions/quality_output/user_pattern/user_pattern_sample10_total_results.json")
    parser.add_argument("--output_path", type=str, default="data/instructions/quality_output/user_pattern/user_pattern_sample10_total_results_parsed.json")
    args = parser.parse_args()
    return args

def parse_output(response):
    """
    Output format:
    ### Output
    - Stage 1: <original instruction> + <user pattern>
    - Stage 2: <user pattern formed in a natural way> 
    - Memory: <Memory about user's preference or routine>
    - User pattern: <user pattern>

    [Example 1]
    ### Input
    Move the book from the living room to the hallway.

    ### Output
    - Stage 1: Move the book from the living room to the hallway. If I'm not done reading the book, I like to keep it in the hallway for easy access.
    - Stage 2: Can you organize this book? I'm not done reading it yet.
    - Memory: Keep books that aren't finished in the hallway for easy access.
    - User pattern: routine
    """
    pattern = r"### Output\n- Stage 1: (.*)\n- Stage 2: (.*)\n- Memory: (.*)\n- User pattern: (.*)"
    match = re.search(pattern, response)
    if match:
        stage1 = match.group(1)
        stage2 = match.group(2)
        memory = match.group(3)
        user_pattern = match.group(4)
        return stage1, stage2, memory, user_pattern

def main(args):
    with open(args.input_path, "r") as f:
        data = json.load(f)

    new_data = []
    for d in data:
        stage1, stage2, memory, user_pattern = parse_output(d['prediction'])
        new_data.append({
            'inst': d['inst'],
            'stage1': stage1,
            'stage2': stage2,
            'memory': memory,
            'user_pattern': user_pattern,
            'org_info': d
        })
        
    with open(args.output_path, "w") as f:
        json.dump(new_data, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    main(args)