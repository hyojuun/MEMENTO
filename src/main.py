#!/usr/bin/env python3
# isort: skip_file

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import sys
import time
import os
import traceback
import json
import shutil
from omegaconf import OmegaConf
from typing import Dict, List, Any, Union
from itertools import islice
from multiprocessing import Pool

# append the path of the
# parent directory
sys.path.append("..")

import hydra
from typing import Dict

from torch import multiprocessing as mp

from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat_baselines.utils.info_dict import extract_scalars_from_info

from habitat_llm.agent.env.evaluation.evaluation_functions import (
    aggregate_measures,
)

from habitat_llm.utils import cprint, setup_config, fix_config


from habitat_llm.agent.env import (
    register_actions,
    register_measures,
    register_sensors,
    remove_visual_sensors,
)

from src.utils.build_memory import build_memory

from src.agent.env import EnvironmentInterface

from src.evaluation import EvaluationRunner

from src.agent.env.dataset import MemoryManagementDatasetV0

def run_scene_process(args):
    config, scene_id, episodes = args
    if config.get("dataset_type", None) == "memory_management":
        new_dataset = MemoryManagementDatasetV0(config=config.habitat.dataset, episodes=episodes)
    else:
        new_dataset = CollaborationDatasetV0(config=config.habitat.dataset, episodes=episodes)

    # 직접 run_planner 호출 (pipe 제거)
    return run_planner(config, new_dataset)


def get_output_file(config, env_interface):
    dataset_file = env_interface.conf.habitat.dataset.data_path.split("/")[-1]
    episode_id = env_interface.env.env.env._env.current_episode.episode_id
    output_file = os.path.join(
        config.paths.results_dir,
        dataset_file,
        "stats",
        f"{episode_id}.json",
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    return output_file


# Function to write data to the CSV file
def write_to_csv(file_name, result_dict):
    # Sort the dictionary by keys
    # Needed to ensure sanity in multi-process operation
    result_dict = dict(sorted(result_dict.items()))
    with open(file_name, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=result_dict.keys())

        # Check if the file is empty (to write headers)
        file.seek(0, 2)
        file_empty = file.tell() == 0
        if file_empty:
            writer.writeheader()

        writer.writerow(result_dict)


def save_exception_message(config, env_interface):
    output_file = get_output_file(config, env_interface)
    exc_string = traceback.format_exc()
    failure_dict = {"success": False, "info": str(exc_string)}
    with open(output_file, "w+") as f:
        f.write(json.dumps(failure_dict))


def save_success_message(config, env_interface, info):
    output_file = get_output_file(config, env_interface)
    failure_dict = {"success": True, "stats": json.dumps(info)}
    with open(output_file, "w+") as f:
        f.write(json.dumps(failure_dict))


# Write the config file into the results folders
def write_config(config):
    dataset_file = config.habitat.dataset.data_path.split("/")[-1]
    output_file = os.path.join(config.paths.results_dir, dataset_file)
    os.makedirs(output_file, exist_ok=True)
    with open(f"{output_file}/config.yaml", "w+") as f:
        f.write(OmegaConf.to_yaml(config))

    # Copy over the RLM config
    planner_configs = []
    suffixes = []
    if "planner" in config.evaluation:
        # Centralized
        if "plan_config" in config.evaluation.planner is not None:
            planner_configs = [config.evaluation.planner.plan_config]
            suffixes = [""]
    else:
        for agent_name in config.evaluation.agents:
            suffixes.append(f"_{agent_name}")
            planner_configs.append(
                config.evaluation.agents[agent_name].planner.plan_config
            )

    for plan_config, suffix_rlm in zip(planner_configs, suffixes):
        if "llm" in plan_config and "serverdir" in plan_config.llm:
            yaml_rlm_path = plan_config.llm.serverdir
            if len(yaml_rlm_path) > 0:
                yaml_rlm_file = f"{yaml_rlm_path}/config.yaml"
                if os.path.isfile(yaml_rlm_file):
                    shutil.copy(
                        yaml_rlm_file, f"{output_file}/config_rlm{suffix_rlm}.yaml"
                    )


# Method to load agent planner from the config
@hydra.main(config_path="./conf")
def run_eval(config):
    fix_config(config)
    # Setup a seed
    # seed = 48212516
    seed = 47668090
    t0 = time.time()
    # Setup config
    config = setup_config(config, seed)
    
    if config.get("dataset_type", None) == "memory_management":
        dataset = MemoryManagementDatasetV0(config.habitat.dataset)
    else:
        dataset = CollaborationDatasetV0(config.habitat.dataset)
    
    write_config(config)

    if config.get("resume", False):
        dataset_file = config.habitat.dataset.data_path.split("/")[-1]
        # stats_dir = os.path.join(config.paths.results_dir, dataset_file, "stats")
        plan_log_dir = os.path.join(
            config.paths.results_dir, dataset_file, "planner-log"
        )

        # Find incomplete episodes
        incomplete_episodes = []
        for episode in dataset.episodes:
            episode_id = episode.episode_id
            # stats_file = os.path.join(stats_dir, f"{episode_id}.json")
            planlog_file = os.path.join(
                plan_log_dir, f"planner-log-episode_{episode_id}_0.json"
            )
            if not os.path.exists(planlog_file):
                incomplete_episodes.append(episode)
        print(
            f"Resuming with {len(incomplete_episodes)} incomplete episodes: {[e.episode_id for e in incomplete_episodes]}"
        )
        # Update dataset with only incomplete episodes
        
        if config.get("dataset_type", None) == "memory_management":
            dataset = MemoryManagementDatasetV0(
                config=config.habitat.dataset, episodes=incomplete_episodes
            )
        else:
            dataset = CollaborationDatasetV0(
                config=config.habitat.dataset, episodes=incomplete_episodes
            )

    # filter episodes by mod for running on multiple nodes
    if config.get("episode_mod_filter", None) is not None:
        rem, mod = config.episode_mod_filter
        episode_subset = [x for x in dataset.episodes if int(x.episode_id) % mod == rem]
        print(f"Mod filter: {rem}, {mod}")
        print(f"Episodes: {[e.episode_id for e in episode_subset]}")
        if config.get("dataset_type", None) == "memory_management":
            dataset = MemoryManagementDatasetV0(
                config=config.habitat.dataset, episodes=episode_subset
            )
        else:
            dataset = CollaborationDatasetV0(
                config=config.habitat.dataset, episodes=episode_subset
            )

    num_episodes = len(dataset.episodes)
    if config.num_proc == 1:
        
        if config.get("episode_indices", None) is not None or config.stage == 1:
            if config.get("resume", False):
                raise ValueError("episode_indices and resume cannot be used together")
            episode_subset = [dataset.episodes[x] for x in config.episode_indices]
            if config.get("dataset_type", None) == "memory_management":
                dataset = MemoryManagementDatasetV0(
                    config=config.habitat.dataset, episodes=episode_subset
                )
            else:
                dataset = CollaborationDatasetV0(
                    config=config.habitat.dataset, episodes=episode_subset
                )
            run_planner(config, dataset)
        
        elif config.stage in [2, 3]:
            # Group episodes by scene_id
            scene_to_episodes = {}
            for ep in dataset.episodes:
                scene_id = ep.scene_id
                if scene_id not in scene_to_episodes:
                    scene_to_episodes[scene_id] = []
                scene_to_episodes[scene_id].append(ep)

            all_stats_episodes: Dict[str, Dict] = {
                str(i): {} for i in range(config.num_runs_per_episode)
            }

            # Process each scene_id sequentially
            for scene_id, episodes in scene_to_episodes.items():
                print(f"Processing scene_id: {scene_id}")

                # Create the dataset for the current scene_id
                if config.get("dataset_type", None) == "memory_management":
                    new_dataset = MemoryManagementDatasetV0(
                        config=config.habitat.dataset, episodes=episodes
                    )
                else:
                    new_dataset = CollaborationDatasetV0(
                        config=config.habitat.dataset, episodes=episodes
                    )

                # Run the planner directly (no multiprocessing)
                run_planner(config, new_dataset)

    else:
        # Process episodes in parallel
        mp_ctx = mp.get_context("forkserver")
        proc_infos = []
        
        if config.stage == 1:
            config.num_proc = min(config.num_proc, num_episodes)
            ochunk_size = num_episodes // config.num_proc
            # Prepare chunked datasets
            chunked_datasets = []
            # TODO: we may want to chunk by scene
            start = 0
            for i in range(config.num_proc):
                chunk_size = ochunk_size
                if i < (num_episodes % config.num_proc):
                    chunk_size += 1
                end = min(start + chunk_size, num_episodes)
                indices = slice(start, end)
                chunked_datasets.append(indices)
                start += chunk_size

            for episode_index_chunk in chunked_datasets:
                episode_subset = dataset.episodes[episode_index_chunk]
                if config.get("dataset_type", None) == "memory_management":
                    new_dataset = MemoryManagementDatasetV0(
                        config=config.habitat.dataset, episodes=episode_subset
                    )
                else:
                    new_dataset = CollaborationDatasetV0(
                        config=config.habitat.dataset, episodes=episode_subset
                    )

                parent_conn, child_conn = mp_ctx.Pipe()
                proc_args = (config, new_dataset, child_conn)
                p = mp_ctx.Process(target=run_planner, args=proc_args)
                p.start()
                proc_infos.append((parent_conn, p))
                print("START PROCESS")

            # Get back info
            all_stats_episodes: Dict[str, Dict] = {
                str(i): {} for i in range(config.num_runs_per_episode)
            }
            for conn, proc in proc_infos:
                stats_episodes = conn.recv()
                for run_id, stats_run in stats_episodes.items():
                    all_stats_episodes[str(run_id)].update(stats_run)
                proc.join()

            all_metrics = aggregate_measures(
                {run_id: aggregate_measures(v) for run_id, v in all_stats_episodes.items()}
            )
            cprint("\n---------------------------------", "blue")
            cprint("Metrics Across All Runs:", "blue")
            for k, v in all_metrics.items():
                cprint(f"{k}: {v:.3f}", "blue")
            cprint("\n---------------------------------", "blue")
            # Write aggregated results across experiment
            write_to_csv(config.paths.end_result_file_path, all_metrics)
        
        elif config.stage in [2, 3]:
            # Group episodes by scene_id
            scene_to_episodes = {}
            for ep in dataset.episodes:
                scene_id = ep.scene_id
                if scene_id not in scene_to_episodes:
                    scene_to_episodes[scene_id] = []
                scene_to_episodes[scene_id].append(ep)

            # Prepare scene arguments for multiprocessing
            scene_args = [(config, sid, eps) for sid, eps in scene_to_episodes.items()]

            # Use multiprocessing Pool to run at most config.num_proc processes at once
            with Pool(processes=config.num_proc) as pool:
                results = pool.map(run_scene_process, scene_args)

            # Aggregate results from all scenes
            all_stats_episodes: Dict[str, Dict] = {
                str(i): {} for i in range(config.num_runs_per_episode)
            }
            for stats_episodes in results:
                try:
                    for run_id, stats_run in stats_episodes.items():
                        all_stats_episodes[str(run_id)].update(stats_run)
                except Exception as e:
                    continue

            all_metrics = aggregate_measures(
                {run_id: aggregate_measures(v) for run_id, v in all_stats_episodes.items()}
            )
            cprint("\n---------------------------------", "blue")
            cprint("Metrics Across All Runs:", "blue")
            for k, v in all_metrics.items():
                cprint(f"{k}: {v:.3f}", "blue")
            cprint("\n---------------------------------", "blue")
            write_to_csv(config.paths.end_result_file_path, all_metrics)
        

    e_t = time.time() - t0
    print(f"Time elapsed since start of experiment: {e_t} seconds.")

    # Calculate and save statistics from epi_result_file_path to mean_result_file_path
    calculate_and_save_statistics(config.paths.epi_result_file_path, config.paths.mean_result_file_path)
    
    if config.get("build_memory", None):
        time.sleep(10) # Wait for output process to finish
        build_memory(config.paths.results_dir, config.habitat.dataset.data_path, dataset)
        
    # epi_result_file_path 에서 같은 episode_id 끼리 묶어서 평균 값, 표준편차, 동일한 episode 개수 계산해서 mean_result_file_path 에 저장


def run_planner(config, dataset: Union[MemoryManagementDatasetV0, CollaborationDatasetV0] = None, conn=None):
    if config == None:
        cprint("Failed to setup config. Exiting", "red")
        return

    # Setup interface with the simulator if the planner depends on it
    if config.env == "habitat":
        # Remove sensors if we are not saving video

        # TODO: have a flag for this, or some check
        keep_rgb = False
        if "use_rgb" in config.evaluation:
            keep_rgb = config.evaluation.use_rgb
        if not config.evaluation.save_video and not keep_rgb:
            remove_visual_sensors(config)

        # TODO: Can we move this inside the EnvironmentInterface?
        # We register the dynamic habitat sensors
        register_sensors(config)
        # We register custom actions
        register_actions(config)
        # We register custom measures
        register_measures(config)

        # Initialize the environment interface for the agent
        env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=False)

        try:
            env_interface.initialize_perception_and_world_graph()
        except Exception:
            print("Error initializing the environment")
            if config.evaluation.log_data:
                save_exception_message(config, env_interface)
    else:
        env_interface = None

    # Instantiate the agent planner (RAG도 여기에서 됨)
    eval_runner: EvaluationRunner = None
    eval_runner = EvaluationRunner(config.evaluation, env_interface)

    # Print the planner
    cprint(f"Successfully constructed the Evaluation Runner!", "green")
    print(eval_runner)

    # Declare observability mode
    cprint(
        f"Partial observability is set to: '{config.world_model.partial_obs}'", "green"
    )

    # Print the agent list
    print("\nAgent List:")
    print(eval_runner.agent_list)

    # Print the agent description
    print("\nAgent Description:")
    print(eval_runner.agent_descriptions)

    # Highlight the mode of operation
    cprint("\n---------------------------------------", "blue")
    cprint(f"Partial Observability: {config.world_model.partial_obs}", "blue")
    cprint("---------------------------------------\n", "blue")

    os.makedirs(config.paths.results_dir, exist_ok=True)

    # Run the planner
    if config.mode == "cli":
        instruction = "Go to the bed" if not config.instruction else config.instruction

        cprint(f'\nExecuting instruction: "{instruction}"', "blue")
        try:
            info = eval_runner.run_instruction(instruction)
        except Exception as e:
            print("An error occurred:", e)

    else:
        stats_episodes: Dict[str, Dict] = {
            str(i): {} for i in range(config.num_runs_per_episode)
        }

        num_episodes = len(env_interface.env.episodes)
        for run_id in range(config.num_runs_per_episode):
            for _ in range(num_episodes):
                # Get episode id
                episode_id = env_interface.env.env.env._env.current_episode.episode_id

                # Get instruction
                instruction = env_interface.env.env.env._env.current_episode.instruction
                print("\n\nEpisode", episode_id)

                try:
                    info = eval_runner.run_instruction(
                        output_name=f"episode_{episode_id}_{run_id}"
                    )

                    info_episode = {
                        "run_id": run_id,
                        "episode_id": episode_id,
                        "instruction": instruction,
                    }
                    stats_keys = {
                        "task_percent_complete",
                        "task_state_success",
                        "sim_step_count",
                        "replanning_count",
                        "runtime",
                    }

                    # add replanning counts to stats_keys as scalars if replanning_count is a dict
                    if "replanning_count" in info and isinstance(
                        info["replanning_count"], dict
                    ):
                        for agent_id, replan_count in info["replanning_count"].items():
                            stats_keys.add(f"replanning_count_{agent_id}")
                            info[f"replanning_count_{agent_id}"] = replan_count

                    stats_episode = extract_scalars_from_info(
                        info, ignore_keys=info.keys() - stats_keys
                    )
                    stats_episodes[str(run_id)][episode_id] = stats_episode
                    
                    # 에러 정보 추가: 에러가 있으면 에러 메시지를, 없으면 False를 추가
                    if "error" in info:
                        error_msg = str(info["error"])
                        info_episode["error"] = error_msg
                        stats_episodes[str(run_id)][episode_id]["error"] = error_msg
                    else:
                        info_episode["error"] = False
                        stats_episodes[str(run_id)][episode_id]["error"] = False
                    
                    cprint("\n---------------------------------", "blue")
                    cprint(f"Metrics For Run {run_id} Episode {episode_id}:", "blue")
                    for k, v in stats_episodes[str(run_id)][episode_id].items():
                        if k == "error" and v:
                            cprint(f"{k}: {v}", "red")  # 에러는 빨간색으로 출력
                        else:
                            cprint(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}", "blue")
                    cprint("\n---------------------------------", "blue")
                    
                    # Log results onto a CSV
                    epi_metrics = stats_episodes[str(run_id)][episode_id] | info_episode
                    if config.evaluation.log_data:
                        save_success_message(config, env_interface, stats_episode)
                    write_to_csv(config.paths.epi_result_file_path, epi_metrics)

                    episode_success = True
                
                except Exception as e:
                    # Start retry logic
                    episode_success = False
                    max_retries = 5
                    retry_count = 0
                    
                    while not episode_success and retry_count < max_retries:
                        retry_count += 1
                        cprint(f"\nRetrying episode {episode_id} (Attempt {retry_count}/{max_retries})...", "yellow")
                    
                        try:
                            # Reset evaluation runner for retry
                            eval_runner.reset()
                            
                            # Re-run the instruction
                            info = eval_runner.run_instruction(
                                output_name=f"episode_{episode_id}_{run_id}_retry{retry_count}"
                            )
                            
                            # Process successful retry
                            info_episode = {
                                "run_id": run_id,
                                "episode_id": episode_id,
                                "instruction": instruction,
                                "retry_count": retry_count,
                            }
                            
                            stats_episode = extract_scalars_from_info(
                                info, ignore_keys=info.keys() - stats_keys
                            )
                            stats_episodes[str(run_id)][episode_id] = stats_episode
                            
                            # Handle error in info dict
                            if "error" in info:
                                error_msg = str(info["error"])
                                info_episode["error"] = error_msg
                                stats_episodes[str(run_id)][episode_id]["error"] = error_msg
                            else:
                                info_episode["error"] = False
                                stats_episodes[str(run_id)][episode_id]["error"] = False
                            
                            cprint("\n---------------------------------", "blue")
                            cprint(f"Metrics For Run {run_id} Episode {episode_id} (Retry {retry_count}):", "blue")
                            for k, v in stats_episodes[str(run_id)][episode_id].items():
                                if k == "error" and v:
                                    cprint(f"{k}: {v}", "red")
                                else:
                                    cprint(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}", "blue")
                            cprint("\n---------------------------------", "blue")
                            
                            # Log results to CSV
                            epi_metrics = stats_episodes[str(run_id)][episode_id] | info_episode
                            if config.evaluation.log_data:
                                save_success_message(config, env_interface, stats_episode)
                            write_to_csv(config.paths.epi_result_file_path, epi_metrics)
                            
                            cprint(f"Successfully completed episode {episode_id} after {retry_count} retries", "green")
                            episode_success = True
                            
                        except Exception as retry_e:
                            traceback.print_exc()
                            cprint(f"Retry {retry_count} failed: {retry_e}", "red")
                    
                    if not episode_success:    
                        # print exception and trace
                        traceback.print_exc()
                        cprint(f"All {max_retries} retries failed for episode {episode_id}", "red")
                        cprint(f"Original error: {e}", "red")

                        # error_metrics
                        error_metrics = {
                            "run_id": run_id,
                            "episode_id": episode_id,
                            "instruction": instruction,
                            "error": str(e),
                            "retry_count": max_retries,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        # error_log_file_path
                        error_log_file_path = os.path.join(
                            os.path.dirname(config.paths.epi_result_file_path),
                            "error_log.csv"
                        )
                        
                        write_to_csv(error_log_file_path, error_metrics)

                        if config.evaluation.log_data:
                            save_exception_message(config, env_interface)

                # Whether episode succedded or all retries failed, we need to move to the next episode
                try:
                    # Reset env_interface (moves onto the next episode in the dataset)
                    env_interface.reset_environment()
                except Exception as e:
                    # print exception and trace
                    traceback.print_exc()
                    print("An error occurred while resetting the env_interface:", e)
                    print("Skipping evaluating episode.")
                    
                    # 환경 리셋 에러도 에러 로그 파일에 기록
                    reset_error_metrics = {
                        "run_id": run_id,
                        "episode_id": episode_id,
                        "instruction": instruction,
                        "error": f"Reset error: {str(e)}",
                        "retry_count": 0,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    error_log_file_path = os.path.join(
                        os.path.dirname(config.paths.epi_result_file_path),
                        "error_log.csv"
                    )
                    write_to_csv(error_log_file_path, reset_error_metrics)
                    
                    if config.evaluation.log_data:
                        save_exception_message(config, env_interface)

                # Reset evaluation runner
                eval_runner.reset()

            # aggregate metrics across the current run.
            run_metrics = aggregate_measures(stats_episodes[str(run_id)])
            cprint("\n---------------------------------", "blue")
            cprint(f"Metrics For Run {run_id}:", "blue")
            for k, v in run_metrics.items():
                cprint(f"{k}: {v:.3f}", "blue")
            cprint("\n---------------------------------", "blue")

            # Write aggregated results across run
            write_to_csv(config.paths.run_result_file_path, run_metrics)

        # aggregate metrics across all runs.
        if conn is None:
            all_metrics = aggregate_measures(
                {run_id: aggregate_measures(v) for run_id, v in stats_episodes.items()}
            )
            cprint("\n---------------------------------", "blue")
            cprint("Metrics Across All Runs:", "blue")
            for k, v in all_metrics.items():
                cprint(f"{k}: {v:.3f}", "blue")
            cprint("\n---------------------------------", "blue")
            # Write aggregated results across experiment
            write_to_csv(config.paths.end_result_file_path, all_metrics)
        else:
            conn.send(stats_episodes)

    env_interface.env.close()
    del env_interface

    if conn is not None:
        # Potentially we may want to send something

        conn.close()


def calculate_and_save_statistics(epi_result_file_path, mean_result_file_path):
    import pandas as pd

    # Read the CSV file into a DataFrame
    df = pd.read_csv(epi_result_file_path)

    # Specify the columns for which to calculate mean and std
    columns_of_interest = ['runtime', 'sim_step_count', 'task_percent_complete', 'task_state_success']

    # Group by 'episode_id' and calculate mean and std for specified columns
    grouped = df.groupby('episode_id')[columns_of_interest].agg(['mean', 'std'])

    # Flatten the MultiIndex columns
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]

    # Save the result to a new CSV file
    grouped.to_csv(mean_result_file_path)


if __name__ == "__main__":
    cprint(
        "\nStart of the main program to run the planner.",
        "blue",
    )

    if len(sys.argv) < 2:
        cprint("Error: Configuration file path is required.", "red")
        sys.exit(1)

    # Run planner
    run_eval()

    cprint(
        "\nEnd of the main program to run the planner.",
        "blue",
    )
