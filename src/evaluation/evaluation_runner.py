#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

"""
This module contains the base EvaluationRunner class and related utilities for running evaluations of LLM-based agents in Habitat environments.
It provides functionality for initializing agents, running episodes, collecting metrics, and storing evaluation results.
The module includes classes for tracking action and state history during evaluation runs.
"""

import copy
import json
import os
import pickle
import time
from typing import Any, Dict, Tuple, List, Optional, Union, TYPE_CHECKING

import attr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from habitat_llm.tools.motor_skills.motor_skill_tool import MotorSkillTool
from habitat_llm.world_model import Entity
from habitat_llm.examples.example_utils import DebugVideoUtil

from src.agent import Agent
from src.agent.env import EnvironmentInterface
from src.planner.planner import Planner
from src.utils import cprint, rollout_print
from src.utils.sim import init_agents
from src.world_model import WorldGraph

from hydra.utils import instantiate


if TYPE_CHECKING:
    from omegaconf import DictConfig
    from src.world_model import WorldGraph
    


@attr.s(auto_attribs=True)
class ActionHistoryElement:
    """
    A class used to represent an element of action history.

    :param action: A tuple representing the action taken of format (Action Type, Action Args).
    :param timestamp: The timestamp at which the action was taken.
    :param agent_uid: The unique identifier of the agent who took the action.
    :param response: The response or feedback received after taking the action.
    :param world_graph: A dictionary mapping agent IDs to their world graph states at this point.
    :param info: Additional information dictionary containing metadata about the action.
    """

    action: tuple
    timestamp: int
    agent_uid: int
    response: str = ""
    world_graph: Dict[int, WorldGraph] = None
    info: dict = attr.ib(factory=dict)

    def to_string(self) -> str:
        """
        Convert the state history element to a string representation.

        :return: A string representation of the agent's state.
        """
        return f"{self.action[0]}[{self.action[1]}]"


@attr.s(auto_attribs=True)
class StateHistoryElement:
    """
    A class used to represent an element of state history.

    :param state: A string representing the state of the agent.
    :param timestamp: The timestamp of the state representation
    :param agent_uid: The unique identifier of the agent in the recorded state.
    """

    state: str
    timestamp: int
    agent_uid: int

    def to_string(self) -> str:
        """
        Convert the state history element to a string representation.

        :return: A string representation of the agent's state.
        """
        return self.state


# Evaluation runner, will go over episodes, run planners and store necessary data.
# Stores an episode, information about the agents and planners and uses them to run through an
# episode and store necessary data.
class EvaluationRunner:
    def __init__(
        self,
        evaluation_runner_config_arg,
        env_interface_arg: EnvironmentInterface,
        dump_world_graph: bool = False,
    ):
        """
        Initialize EvaluationRunner

        :param evaluation_runner_config_arg: The experiment configuration, including config of the agents and planners.
        :param env_interface_arg: The environment instance
        :param dump_world_graph: Whether to dump the world graph to a file.
        """
        self.env_interface = env_interface_arg
        self.evaluation_runner_config = evaluation_runner_config_arg
        self.TRUNCATE_LENGTH = self.evaluation_runner_config.truncate_length

        dataset_file = self.env_interface.conf.habitat.dataset.data_path.split("/")[-1]
        results_dir = self.env_interface.conf.paths.results_dir
        self.output_dir = f"{results_dir}/{dataset_file}/"
        os.makedirs(self.output_dir, exist_ok=True)

        # Declare container to store agent positions
        self.agent_positions: List[Any] = []
        self.object_nodes: List[Entity] = []

        # Declare a container for storing unique agents
        self.agents: Dict[int, Agent] = {}

        self.episode_filename = ""
        self.current_instruction = ""

        # Initialize the agents
        self.__initialize_agents()

        self.planner: Union[Dict[int, Planner], Planner] = {}
        self._initialize_planners()

        # Initialize the debug video util
        self.dvu = DebugVideoUtil(self.env_interface, self.output_dir)
        self._write_out_world_graph: bool = dump_world_graph
        self._world_graph_write_out_frequency = 5

    def _initialize_planners(self) -> None:
        """
        Initialize the centralized planner based on the evaluation runner configuration.
        Sets up the planner with appropriate agents and configures special planning modes
        if specified.
        """
        planner_conf = self.evaluation_runner_config.planner
        planner = instantiate(planner_conf)
        self.planner: Planner = planner(env_interface=self.env_interface)

        # Set just robot agent to the planner
        # we can call only agent_0_~ by this
        self.planner.agents = [self.agents[0]]
        if (
            "plan_config" in planner_conf
            and "planning_mode" in planner_conf.plan_config
            and planner_conf.plan_config.planning_mode == "st"
        ):
            for agent in self.planner.agents:
                for tool in agent.tools.values():
                    if isinstance(tool, MotorSkillTool):
                        tool.error_mode = "st"

    # Method to initialize the agents based in the config
    def __initialize_agents(self) -> None:
        """
        Initialize agents based on config.
        """
        agents = init_agents(self.evaluation_runner_config.agents, self.env_interface)
        for agent in agents:
            self.agents[agent.uid] = agent
            agent._dry_run = self.env_interface._dry_run
            cprint(f"successfully added agent with UID : {agent.uid}", "green")
        print("finished initializing agents!")
        print(self.agents)

    # Method to print the object
    def __str__(self) -> str:
        """
        Return string with state of the evaluator
        """
        planner_type = type(self.planner)
        out = f"Centralized Planner: {planner_type}\n"
        out += f"Number of Agents: {len(self.agents)}"
        return out

    @property
    def agent_list(self) -> str:
        """Returns a string listing the agent's uid"""
        return str([agent.uid for agent in self.agents.values()])

    @property
    def tool_list(self) -> List[str]:
        """Returns a list of unique tool names across all agents.

        :return: List of tool names as strings
        """
        tool_set = set()
        for agent in self.agents.values():
            for tool in agent.tools.values():
                tool_set.add(tool.name)

        return list(tool_set)

    def reset(self) -> None:
        """Reset metrics and stats to be ready for the next episode."""

        # Clear the frames to make sure that
        # video for next episode does no have frames from previous run
        self.dvu.frames.clear()

        # Clear containers used for top-down video generation
        self.agent_positions.clear()
        self.object_nodes.clear()

        # Reset filenames
        self.episode_filename = ""
        self.current_instruction = ""

        # Reset planners and the agents owned by the planners
        # This will also reset skills owned by the agents to
        # make eval runner ready for next episode
        self.reset_planners()

    @property
    def agent_descriptions(self) -> str:
        """Returns a string listing the descriptions of all agents

        :return: A concatenated string of all agent descriptions
        """
        out = ""
        for agent in self.agents.values():
            out += agent.agent_description
        return out

    def _update_td(self, frame, ax) -> None:
        """
        Function to update the top down plot for each robot position
        and detected objects over time.

        :param frame: Current animation frame number
        :param ax: Matplotlib axis object to draw on
        """
        # Clear the current plot
        ax.clear()

        # Extract x and y positions for the current frame
        x = [position[0] for position in self.agent_positions[: frame + 1]]
        y = [position[2] for position in self.agent_positions[: frame + 1]]

        # Extract object x and y
        x_obj = [obj.properties["translation"][0] for obj in self.object_nodes[frame]]
        y_obj = [obj.properties["translation"][2] for obj in self.object_nodes[frame]]
        names = [obj.name for obj in self.object_nodes[frame]]

        # Plot the robot's path
        ax.plot(x, y, marker=".", linestyle="-")

        # Plot the objects path
        ax.scatter(x_obj, y_obj, marker="*", color="red")

        # Add text near each point
        for _, txt in enumerate(zip(x_obj, y_obj, names)):
            ax.text(txt[0], txt[1], txt[2], color="red", ha="right", va="bottom")

        # Set labels and title
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_title("Robot Movement Over Time")

        # Set axis limits
        ax.set_xlim(-25, 25)
        ax.set_ylim(-25, 25)

        # Add grid
        ax.grid(True)

        # Set aspect ratio to be equal
        ax.set_aspect("equal")

    def _store_for_top_down_viz(self, agent_uid: Optional[int] = None) -> None:
        """
        Stores agent position and world graph object data for top-down visualization.

        :param agent_uid: Optional ID of agent whose perspective to use. If None, uses full observability.
        """
        world_graph = None
        if agent_uid is not None:
            world_graph = self.env_interface.world_graph[agent_uid]
        else:
            print(
                "No agent_uid provided. Code will generate top-down visualization from full-observability perspective"
            )
            world_graph = self.env_interface.full_world_graph
        sim = self.env_interface.sim
        self.agent_positions.append(
            sim.agents_mgr[agent_uid].articulated_agent.base_pos
        )

        self.object_nodes.append(world_graph.get_all_objects())

    def _log_planner_data(self, planner_infos: List[Dict[str, Any]]) -> None:
        """
        Logs planner data including prompts, traces and other info to files.

        :param planner_infos: List of dictionaries containing planner information at each step
        """
        # Print logging
        print("\nLogging planner data ...")

        # Log the latest prompts and traces
        for agent in self.agents.values():
            # -----------------------------------------------
            # Save prompts
            # Contains special tokens and few shot examples
            # -----------------------------------------------
            if "prompts" in planner_infos[-1]:
                file_path_prompts = os.path.join(
                    self.output_dir,
                    "prompts",
                    str(agent.uid),
                    f"prompt-{self.episode_filename}-{str(agent.uid)}.txt",
                )

                os.makedirs(os.path.dirname(file_path_prompts), exist_ok=True)

                with open(file_path_prompts, "w") as file:
                    file.write(planner_infos[-1]["prompts"][agent.uid])

            # -----------------------------------------------
            # Save traces
            # Skips special tokens and few shot examples
            # -----------------------------------------------
            if "traces" in planner_infos[-1]:
                file_path_traces = os.path.join(
                    self.output_dir,
                    "traces",
                    str(agent.uid),
                    f"trace-{self.episode_filename}-{str(agent.uid)}.txt",
                )

                os.makedirs(os.path.dirname(file_path_traces), exist_ok=True)

                with open(file_path_traces, "w") as file:
                    file.write(planner_infos[-1]["traces"][agent.uid])

        # Log other info from planner
        file_path_json = os.path.join(
            self.output_dir,
            "planner-log",
            f"planner-log-{self.episode_filename}.json",
        )

        # write the agents_to_actions (the plan)
        if "actions_per_agent" in planner_infos[-1]:
            actions_per_agent_path = os.path.join(
                self.output_dir,
                f"plan/{self.episode_filename}.txt",
            )
            os.makedirs(os.path.dirname(actions_per_agent_path), exist_ok=True)
            with open(actions_per_agent_path, "w") as file:
                file.write(str(planner_infos[-1]["actions_per_agent"]))

        # Make directory if it doesn't exists already
        os.makedirs(os.path.dirname(file_path_json), exist_ok=True)

        # Dictionary to store final log
        planner_log: Dict[str, Any] = {
            "task": self.current_instruction,
            "steps": [],
        }

        # Declare keys to exclude
        keys_to_exclude = ["prompts", "traces", "print", "print_no_tags"]

        # Add planner info at each step
        for i, planner_info in enumerate(planner_infos):
            step_info = {
                k: v
                for k, v in sorted(planner_info.items())
                if k not in keys_to_exclude
            }
            step_info["log_index"] = i
            planner_log["steps"].append(step_info)

        with open(file_path_json, "w+") as file:
            file.write(json.dumps(planner_log, indent=4))

        print("Successfully logged planner data!")
        if self.evaluation_runner_config.log_detailed_traces:
            self._save_detailed_traces()

    def _save_detailed_traces(self) -> None:
        """
        Save detailed traces to a pickle file containing instruction, action history and state history.
        """
        for actions in self.env_interface.agent_action_history.values():
            # don't check the last action because if you hit the max sim step count no result will be logged
            for action in actions[:-1]:
                if action.response in [None, ""] and action.action[0] != "Done":
                    action_history_string = "\n".join(
                        [f"{a.action}: {a.response}" for a in actions]
                    )
                    raise ValueError(
                        f"Agent {action.agent_uid} has a null response on {action.action}: Action history:\n{action_history_string}"
                    )

        file_path_detailed_trace = os.path.join(
            self.output_dir,
            "detailed_traces",
            f"detailed_trace-{self.episode_filename}.pkl",
        )
        result = {
            "instruction": self.current_instruction,
            "action_history": self.env_interface.agent_action_history,
            "state_history": self.env_interface.agent_state_history,
        }

        os.makedirs(os.path.dirname(file_path_detailed_trace), exist_ok=True)

        with open(file_path_detailed_trace, "wb") as file:
            pickle.dump(result, file)

    def _make_td_video(self) -> None:
        """
        Creates a top-down video visualization of the episode.

        :param instruction: Task instruction being executed
        """
        os.makedirs(f"{self.output_dir}/videos", exist_ok=True)
        td_video_name = f"{self.output_dir}/videos/video-td-{self.episode_filename}.mp4"

        # Create a figure and axis for the plot
        fig, ax = plt.subplots()

        # Set the number of frames in the animation
        num_frames = len(self.agent_positions)

        # Create the animation
        animation = FuncAnimation(
            fig, self._update_td, fargs=(ax,), frames=num_frames, repeat=False
        )

        # Save the animation as a video file (e.g., .mp4)
        animation.save(td_video_name, writer="ffmpeg", fps=30)

    def initialize_instruction_metadata(
        self, instruction: str, output_name: str
    ) -> None:
        """
        Start folders where the outputs will be stored.

        :param instruction: The natural language instruction for the task to be executed. If None, uses the instruction from the current episode.
        :param output_name: Name to use for output files. If empty string, generates name from instruction text.
        """
        if instruction is None:
            # Get the instruction from the episode
            self.current_instruction = (
                self.env_interface.env.env.env._env.current_episode.instruction
            )
        else:
            self.current_instruction = instruction
        if self.evaluation_runner_config.do_print:
            cprint("Instruction:", "yellow")
            print(self.current_instruction + "\n")
        # Make hyphenated instruction for creating a filename
        if len(output_name) == 0:
            self.episode_filename = self.current_instruction.replace(" ", "-")[
                :-1
            ].lower()
        else:
            self.episode_filename = output_name
        # check if name is too long, truncate to be system-friendly
        if len(self.episode_filename) > self.TRUNCATE_LENGTH:
            self.episode_filename = self.episode_filename[: self.TRUNCATE_LENGTH]

    def get_low_level_actions(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph: Dict[int, "WorldGraph"],
    ) -> Tuple[Dict[int, Any], Dict[str, Any], bool]:
        """
        Given a set of observations, gets a vector of low level actions for all agents.

        :param instruction: Natural language instruction describing the task to execute
        :param observations: Dictionary of observations from the environment
        :param world_graph: Dictionary mapping agent IDs to their world graph states

        :return: Tuple containing:
            - List of low level action dictionaries for each agent
            - Dictionary with planner info about high level actions
            - Boolean indicating whether execution should end
        """
        low_level_actions, planner_info, should_end = self.planner.get_next_action(
            instruction, observations, world_graph
        )
        return low_level_actions, planner_info, should_end

    def reset_planners(self) -> None:
        """Reset the centralized planner to prepare for a new episode."""
        assert isinstance(self.planner, Planner)
        self.planner.reset()

    def update_agent_state_history(self, planner_info: Dict[str, Any]) -> None:
        """
        Updates the state history stored in env_interface based on planner info.
        This includes logging states like "standing", "walking", "picking X", "placing on Y" etc.

        :param planner_info: Dictionary containing planner state information
        """
        # # Update the agent states in environment interface
        if "agent_states" in planner_info:
            for agent_uid in planner_info["agent_states"]:
                agent_state_at_t = planner_info["agent_states"][agent_uid]
                if len(self.env_interface.agent_state_history[agent_uid]) > 0:
                    agent_state_at_t_minus_1 = self.env_interface.agent_state_history[
                        agent_uid
                    ][-1]
                    if agent_state_at_t != agent_state_at_t_minus_1.state:
                        self.env_interface.agent_state_history[agent_uid].append(
                            StateHistoryElement(
                                agent_state_at_t,
                                planner_info["sim_step_count"],
                                agent_uid=agent_uid,
                            )
                        )
                else:
                    self.env_interface.agent_state_history[agent_uid].append(
                        StateHistoryElement(
                            agent_state_at_t,
                            planner_info["sim_step_count"],
                            agent_uid=agent_uid,
                        )
                    )

    def update_agent_action_history(self, planner_info: Dict[str, Any]) -> None:
        """
        Updates the actions history stored in env_interface based on planner info.
        This includes logging actions like "Navigate[object_id]", "Pick[object_id]" etc.

        :param planner_info: Dictionary containing planner action information
        """
        if "replan_required" not in planner_info:
            return
        # Update the agent states in environment interface
        for agent_id, value in planner_info["replanned"].items():
            if value:
                # An action must be returned if the planner replans
                assert agent_id in planner_info["high_level_actions"]
                action_history_object = ActionHistoryElement(
                    action=planner_info["high_level_actions"][agent_id],
                    timestamp=planner_info["sim_step_count"],
                    agent_uid=agent_id,
                    world_graph=copy.deepcopy(self.env_interface.world_graph),
                    info={
                        "planner_info": planner_info,
                        "log_time": time.time(),
                    },
                )

                self.env_interface.agent_action_history[agent_id].append(
                    action_history_object
                )

        # add responses the last logged action, this means the planner will replan at the next step
        if "responses" in planner_info and any(planner_info["responses"].values()):
            for agent_id, response in planner_info["responses"].items():
                # empty string response does not mean the action is over
                # skip adding the response
                if response == "":
                    continue
                # There should have been an action logged if there is a response
                assert len(self.env_interface.agent_action_history[agent_id]) > 0
                self.env_interface.agent_action_history[agent_id][
                    -1
                ].response = response
                for ah in self.env_interface.agent_action_history[agent_id]:
                    if ah.response is None or len(ah.response) == 0:
                        raise ValueError(
                            f"Agent {agent_id} has a null response on {ah.action}"
                        )

    def run_instruction(
        self, instruction: Optional[str] = None, output_name: str = ""
    ) -> Dict[str, Any]:
        """
        Runs a single instruction through the planner, taking steps until the task is done.
        Stores the information using the provided output name.

        :param instruction: Optional instruction to execute. If None, uses episode instruction.
        :param output_name: Name to use for output files. If empty, derives from instruction.

        :return: Dictionary containing execution information and metrics
        """
        # Log start time
        t_0 = time.time()

        # Counter to count iterations
        # of this loop, as sim step dont increase
        # for perception tools
        total_step_count = 1

        # Reset planners and the agents owned by the planners
        # This will also reset skills owned by the agents to
        # make eval runner ready for next episode
        self.reset_planners()

        # Initialize metadata
        self.initialize_instruction_metadata(instruction, output_name)
        # Initialize sensor observations
        observations = self.env_interface.get_observations()

        # Dictionary to store info about episode execution
        # Set default metrics incase the motor skills are never called
        # and episode ends
        info = {
            "task_percent_complete": 0.0,
            "task_state_success": 0.0,
            "total_step_count": total_step_count,
            "num_steps": 0.0,
        }

        # List to store planner logs at each step
        planner_infos = []
        planner_info: Dict[str, Any] = {}
        low_level_actions: List[Dict[str, Any]] = []
        should_end = False

        # Plan until required
        while not should_end:
            # Print the llm response
            if (
                "print" in planner_info
                and len(planner_info["print"])
                and self.evaluation_runner_config.do_print
            ):
                rollout_print(planner_info["print"])
            # Execute low level actions
            if len(low_level_actions) > 0:
                obs, reward, done, info = self.env_interface.step(low_level_actions)
                # Refresh observations
                observations = self.env_interface.parse_observations(obs)
                if self.evaluation_runner_config.save_video:
                    # Store third person frames for generating video
                    self.dvu._store_for_video(
                        observations, planner_info["high_level_actions"]
                    )

            # Get next low level actions
            low_level_actions, planner_info, should_end = self.get_low_level_actions(
                self.current_instruction, observations, self.env_interface.world_graph
            )

            # We terminate the episode if this loop gets stuck
            curr_env = self.env_interface.env.env.env._env

            if total_step_count > curr_env._max_episode_steps:
                should_end = True

            measure_names = [
                "auto_eval_proposition_tracker",
                "task_constraint_validation",
                "task_percent_complete",
                "task_state_success",
                "task_evaluation_log",
                "task_explanation",
            ]
            measures_to_log = [
                "task_percent_complete",
                "task_state_success",
                "task_explanation",
            ]
            if should_end:
                measures = curr_env.task.measurements.measures
                for measure_name in measure_names:
                    measures[measure_name].update_metric(
                        task=curr_env.task, episode=curr_env.current_episode
                    )
                for measure_name in measure_names:
                    if measure_name in info:
                        info[measure_name] = measures[measure_name].get_metric()

            # Add performance stats and to planner_info
            planner_info["stats"] = {
                info_name: info[info_name]
                for info_name in measures_to_log
                if info_name in info
            }

            # Add step count to planner_info
            planner_info["total_step_count"] = total_step_count
            planner_info["sim_step_count"] = info["num_steps"]

            # Add world description to planner_info
            # on every replanning step and at the end of planning
            if (
                "replan_required" in planner_info
                and planner_info["replan_required"]
                and any(planner_info["replan_required"].values())
            ) or should_end:
                planner_info["curr_graph"] = {
                    agent_id: self.env_interface.world_graph[agent_id].get_world_descr()
                    for agent_id in range(len(self.agents))
                }

            # Update agent state and action history
            copy_planner_info = copy.deepcopy(planner_info)
            self.update_agent_state_history(copy_planner_info)
            self.update_agent_action_history(copy_planner_info)

            # Append planner info to history
            planner_infos.append(copy_planner_info)

            # Increment while loop step count
            total_step_count += 1

            if (
                self._write_out_world_graph
                and total_step_count % self._world_graph_write_out_frequency == 0
            ):
                # dump the world-graph somewhere to compare
                for agent_id in self.env_interface.world_graph:
                    filename = f"{self.env_interface.env.env.env._env.current_episode.episode_id}_wg_agent_{agent_id}_iter_{total_step_count}.txt"
                    filepath = os.path.join(self.output_dir, filename)
                    with open(filepath, "w") as f:
                        self.env_interface.world_graph[agent_id].display_hierarchy(
                            file_handle=f
                        )
                    print(f"WG written to:\n{filepath}")

        # Print
        if (
            "print" in planner_info
            and len(planner_info["print"])
            and self.evaluation_runner_config.do_print
        ):
            rollout_print(planner_info["print"])

        # Make video
        if self.evaluation_runner_config.save_video:
            self.dvu._make_video(play=False, postfix=self.episode_filename)

        # Log planner information per step
        self._log_planner_data(planner_infos)

        # Log overall time
        t_runtime = time.time() - t_0
        info["runtime"] = t_runtime

        # Merge dictionaries
        info |= planner_info

        return info