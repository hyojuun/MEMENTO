import argparse
import asyncio
import json
import yaml
import os
import random
from copy import deepcopy
import gzip
import pickle

import numpy as np
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from langchain_community.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

TOTAL_COST = 0  # making this a global variable, be aware this may lead to issues in concurrent scenarios
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="gpt-3.5-turbo or gpt-4")
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_key", type=str, default=None)
    parser.add_argument("--save_instances", action="store_true", help="If you want to save the instances, set this argument")
    parser.add_argument("--start_idx", type=int, default=0, help="If you want to start from a specific index, set this argument")
    parser.add_argument("--save_dir", type=str, required=True, help="It should be a NEW DIRECTORY. Please do not use an existing")
    parser.add_argument("--num_sample", type=int, default=None, help="If you want to test your code by sampling a small number of data, you can set this argument.")
    ## generate args ##
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--stop_sequence", type=str, nargs='+', default=None)
    parser.add_argument("--sampling_num", type=int, default=1, help="The number of samples to generate per instance")
    args = parser.parse_args()
    if args.num_sample:
        args.save_dir = args.save_dir + f"_sample{args.num_sample}"
        
        
    os.makedirs(args.save_dir, exist_ok=True)
    return args

## TODO : change this function to load the prompt from your own file ##
def load_prompt(args):
    with open(args.prompt, "r", encoding="UTF-8") as f:
        prompt = yaml.load(f, Loader=yaml.FullLoader)[args.prompt_key]
    return prompt


## TODO : change this function to prepare the model input from your own data ##
def prepare_model_input(prompt:str, data_path:str):
    '''
        input : prompt, data_path (str)
        output : all_model_data (list of dict)
    '''
    with open(data_path, "r", encoding="UTF-8") as f:
        data = json.load(f)

    all_model_data = []
    for d in tqdm(data):
        input_temp = deepcopy(d)
        ## TODO : change this code to prepare the model input from your own data ##
        input_temp['org_instruction'] = d['inst']
        input_temp['model_input'] = prompt.format(**input_temp)
        all_model_data.append(input_temp)

    return all_model_data


def load_and_prepare_data(args):
    prompt = load_prompt(args)
    print("Preparing model inputs...")
    all_model_data = prepare_model_input(
        prompt, args.input_path)
    return all_model_data


def sample_indices(all_model_inputs, num_sample):
    random.seed(0)
    cand_indices = list(range(len(all_model_inputs)))
    sampled_indices = random.sample(cand_indices, num_sample)
    return sampled_indices


def filter_data(all_model_data, num_sample):
    if num_sample:
        sampled_indices = sample_indices(all_model_data, num_sample)
        all_model_data = [all_model_data[i] for i in sampled_indices]
    return all_model_data


async def async_generate(args, llm, model_data, idx):
    global TOTAL_COST
    system_message = SystemMessage(content=model_data['model_input'])
    # human_message = HumanMessage(content=model_input) # if you need it
    while True:
        try:
            response = await llm.agenerate([[system_message]])
            # response = await llm.agenerate([[system_message, human_message]]) # if you need it
            token_used = response.llm_output['token_usage']['total_tokens']
            
            ## TODO : Change the cost calculation code if you want to use a different model ##
            if args.model_name == "gpt-3.5-turbo":
                TOTAL_COST += token_used / 1000 * 0.002
            elif args.model_name == "gpt-4":
                TOTAL_COST += token_used / 1000 * 0.06 
            # print(idx, TOTAL_COST)
            break
        
        except Exception as e:
            print(f"Exception occurred: {e}")
            response = None

    ## TODO : change this code if you want to save it in a different way ##
    result = deepcopy(model_data) 
    result['prediction'] = response.generations[0][0].text
    ## TODO : Add some post-processing if you want ##
    
    if args.save_instances:
        with open(os.path.join(args.save_dir, f"{idx}.json"), "w", encoding='UTF-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
    return result


async def generate_concurrently(args, all_model_data, start_idx):
    llm = ChatOpenAI(model_name=args.model_name,  # 'gpt-3.5-turbo' or 'gpt-4'
                     temperature=args.temperature, 
                     max_tokens=args.max_tokens,
                     max_retries=100,
                     top_p=args.top_p,
                     frequency_penalty=args.frequency_penalty,
                     stop=args.stop_sequence,
                     n=args.sampling_num,
                     )
    tasks = [async_generate(args, llm, model_data, i+start_idx)
             for i, model_data in enumerate(all_model_data)]
    
    return await tqdm_asyncio.gather(*tasks)


async def main(args):
    all_model_data = load_and_prepare_data(args)
    all_model_data = filter_data(all_model_data, args.num_sample)

    # Check if the save_dir exists
    if args.save_instances and os.path.exists(args.save_dir):
        print("The save_dir already exists. Please change the save_dir.")

    if args.save_instances:
        os.makedirs(args.save_dir, exist_ok=True)
        
    all_results = []
    if len(all_model_data) - args.start_idx > 300:
        for start_idx in tqdm(range(args.start_idx, len(all_model_data), 300)):
            cur_model_data = all_model_data[start_idx:start_idx+300]
            all_results.extend(await generate_concurrently(args, cur_model_data, start_idx))
    else:
        all_results = await generate_concurrently(args, all_model_data, args.start_idx)

    total_result_path = args.save_dir + "_total_results.json"
    with open(os.path.join(total_result_path), "w", encoding='UTF-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
        
if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
