## Introduction

Habitat 3.0 simulator 상에서 LLM 이용해 collaborative tasks를 수행하는 PARTNR benchmark 코드 기반으로 작성하였음.

- habitat 3.0 simulator
  - habitat-lab (habitat simulator wrapper, run with .yaml)
  - habitat-sim (simulator, written in C++)
- habitat_llm: partnr-planner code에서 작성한 LLM을 이용해 simulation에서 task 수행하기 위한 codes

## Installation

For installation, refer to [INSTALLATION.md](INSTALLATION.md)
그대로 따라하면 잘 됩니다. simulation video rendering 등등 위해선 추가 package 설치 필요할 수 있음.

## Quick Start

**Decentralized Multi Agent React Summary**

```bash
python -m habitat_llm.examples.planner_demo --config-name baselines/decentralized_zero_shot_react_summary.yaml \
    habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val_mini.json.gz" \
    evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_1.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine=meta-llama/Meta-Llama-3-8B-Instruct \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.engine=meta-llama/Meta-Llama-3-8B-Instruct
```

**Single Agent React Summary**

```bash
python -m habitat_llm.examples.planner_demo --config-name baselines/single_agent_zero_shot_react_summary.yaml \
    habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val_mini.json.gz" \
    evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine=meta-llama/Meta-Llama-3-8B-Instruct
```

## Architecture

<p align="center">
  <img src="docs/Overview.png" width="65%"/>
</p>

대략적으로 위 그림과 같은 구조로 돌아감.

- Setting은 Hydra Configure를 사용해야 함
  - habitat_lab과 habitat_llm 모두 설정을 yaml 파일들로 설정함
    ⇒ habitat_llm/conf baselines를 보면 yaml 설정하는 방법 감 오실거에요
- 기본적인 running 방식

  1. Hydra로 configure 정리하고, Agent 단위 설정(ex. sensor, action 등록 등)을 마친 후 **_EnvironmentInterface_**를 initialize 합니다.
  2. PARTNR의 기본 setting인 Decentralized, Centralized 설정에 따른 **\*EvaluationRunner** (ex. DecentralizedEvaluationRunner, CentralizedEvaluationRunner)\*를 선언합니다.

     따라서, EvaluationRunner 새롭게 정의해야 함.

  3. **_EvaluationRunner_**를 통해 evaluation을 수행합니다. 여기서 evaluation이라 함은, episode에 대한 agent의 수행을 평가하는 evaluation입니다. **_EvaluationRunner_**에선 evalution을 돌리기 위한 것들을 선언합니다. ex. **_WorldGraph_** (GT or Concept)는 어떻게 할지, **_Perception_**은 어떻게 할지 (ex. world*model / Observability (Fully or Partial)에 따라 다름), \*\*\_Planner*\*\* 등
  4. Evalution을 수행하게 되면, configure로 설정한 sensors로 부터 **_EnvironmentInterface_**가 정보를 받아 **_EvaluationRunner_**에 계속 넘겨줍니다. **_Perception_**이 sensor 정보를 받아 preprocessing을 수행합니다. 여기서 world_graph에 추가할지 말지를 결정합니다.

     GT의 경우완 다르게, ConceptGraphs의 경우는 **_DynamicWorldGraph_**로 init됩니다. 따라서, 실행 중간 계속 새롭게 observe한 것을 Entity라는 base class로 정의된 다양한 classes(ex. Room, Object …) 중 하나를 골라 node로 추가합니다.

     이 worldgraph가 연구의 memory scene graph에 해당한다고 할 수 있고, 이부분을 develop 해야 합니다.

  5. Planner는 기본적으로 CurrentWorldGraph, Observation, CurrentInstruction 3가지를 받아 action을 수행합니다. **_action_**은 low-level action과 high-level action으로 구성됩니다.

     맨 처음 instruction을 받으면 thought을 통해 가장 처음 수행할 instruction을 결정합니다. 만약, task를 성공했다면 replan을 통해 다음 instruction을 planning 합니다. task를 수행하기 위해 Agent는 low-level action (explore 등)은 계속(step을 통해)해서 진행합니다.

- Episode 생성 방식 (PARTNR)

  1. instruction preset을 만들고, preset 기반해 scene으로 부터 augmentation합니다.
  2. instruction generation 수행

     HSSD metadata file이 이미 있어, 이걸 이용해서 dataframe을 만들고, 그걸 참고해서 templated prompt를 구성합니다. 이 prompt를 LLM에 전달해 Instruction을 생성합니다.

  3. instruction parsing 수행
  4. parsed instruction 기반 filtering 수행
  5. episode generation 수행
     따라서, 우리가 원하는 episode를 생성하려면
  6. HSSD 속 pre-set option에서 dynamic하게 추가 정보만 부여
  7. HSSD 변형해, 원하는 setting을 만들기 (ex. 여러 개의 컵을 붙여 놓는다던지 등의 상황, 키 두개를 같이 둔다던지의 상황)
  8. 다른 dataset 직접 build
     해야할 듯 해요.

## Development Details

주요 class 들이 어떻게 구성되어 있는지 간략 요약합니다. 보시면 이해 쉽게 되실거에요.

1. EvaluationRunner (ex. DecentralizedEvaluationRunner)

    ```python
    class DecentralizedEvaluationRunner(EvaluationRunner):
        def __init__(self, evaluation_runner_config_arg, env_arg: EnvironmentInterface):
            # Call EvaluationRunner class constructor
            super().__init__(evaluation_runner_config_arg, env_arg)

        def run_instruction(self, instruction=None, output_name=""):
            # Runs a single instruction through the planner, taking steps until the task is done. Stores the information in output name
            
            ...
            
            while not should_end:
                
                ...
                
                # Execute low level actions
                if len(low_level_actions) > 0:
                    **obs, reward, done, info = self.env_interface.step(low_level_actions)**
                    # Refresh observations
                    observations = self.env_interface.parse_observations(obs)

                    if self.evaluation_runner_config.save_video:
                        # Store third person frames for generating video
                        self.dvu._store_for_video(
                            observations, planner_info["high_level_actions"]
                        )

                # Get next low level actions
                **low_level_actions, planner_info, should_end = self.get_low_level_actions(
                    self.current_instruction, observations, self.env_interface.world_graph
                )**

   				...

                # Add world description to planner_info
                # on every replanning step and at the end of planning
                if (
                    planner_info["replan_required"]
                    and any(planner_info["replan_required"].values())
                ) or should_end:
                    planner_info["curr_graph"] = {
                        agent_id: self.env_interface.world_graph[agent_id].get_world_descr(
                            is_human_wg=int(agent_id) == self.env_interface.human_agent_uid
                        )
                    }
   			    
                ...

        def get_low_level_actions(self, instruction, observations, world_graph):
            """
            Given a set of observations, gets a vector of low level actions, an info dictionary and a boolean indicating that the run should end.
            """
   			
            ...

            # Loop through all available planners
            for planner in self.planner.values():
                # Get next action for this planner
                (
                    this_planner_low_level_actions,
                    this_planner_info,
                    this_planner_is_done,
                ) = **planner.get_next_action(instruction, observations, world_graph)**
                # Update the output dictionary with planned low level actions
                low_level_actions.update(this_planner_low_level_actions)

            ...

    ```

2. Planner (LLMPlanner)

    ```python
    class LLMPlanner(Planner):
        """
        High level planner policy used by agents to decide high level actions given task description and state of the world
        """
        def __init__(self, plan_config: "DictConfig", env_interface: "EnvironmentInterface"):
   	        # Initialize the LLMPlanner.

        def reset(self):
   		    # Reset the planner state.

   	    def build_tool_grammar(self, world_graph: "WorldGraph") -> str:
   		    """
   		    This method builds a grammar that accepts all valid tool calls based a world graph
            The grammar is specified in the EBNF grammar description format
   		    """

        def build_response_grammar(self, world_graph: "WorldGraph") -> str:
   		    # Build a grammar that accepts all valid responses based on a world graph.

        def prepare_prompt(
           self, input_instruction: str, world_graph: "WorldGraph"
        ) -> Tuple[str, Dict[str, Any]]:
            """
            Prepare the prompt for the LLM. Used to update the prompt based on the instruction and the world_graph
            """

   	    def get_last_agent_states(self) -> Dict[int, str]:
   		    # Get the last state descriptions for all agents.

        def get_last_agent_positions(self) -> Dict[str, Any]:
            # Get the last positions for all agents.

        def get_next_action(
            self,
            instruction: str,
            observations: Dict[str, Any],
            world_graph: Dict[int, "WorldGraph"],
            verbose: bool = False,
        ) -> Tuple[Dict[int, Any], Dict[str, Any], bool]:
            # Get the next low-level action to execute.

            # Early return if planner is already done
            if self.is_done:
   		        # Early return
   		        
                ...
                
                return {}, planner_info, self.is_done

            if self.curr_prompt == "":
                # Prepare prompts
                self.curr_prompt, self.params = self.prepare_prompt(
                   instruction, world_graph[self._agents[0].uid]
                )
                self.curr_obj_states = get_objects_descr(
                    world_graph[self._agents[0].uid],
                    self._agents[0].uid,
                    include_room_name=True,
                    add_state_info=self.planner_config.objects_response_include_states,
                    centralized=self.planner_config.centralized,
                )

   				...

   		    # 만약, task 성공해서 replan 해야 하는 경우
            if self.replan_required:
                planner_info["replanned"] = {agent.uid: True for agent in self.agents}
                # Generate response
                if self.planner_config.get("constrained_generation", False):
                    llm_response = self.llm.generate(
                        self.curr_prompt,
                        self.stopword,
                        generation_args={
                            "grammar_definition": self.build_response_grammar(
                                world_graph[self._agents[0].uid]
                            )
                        },
                    )
                else:
                    llm_response = self.llm.generate(self.curr_prompt, self.stopword)

                # Format the response
                # This removes extra text followed by end expression when needed.
                llm_response = self.format_response(llm_response, self.end_expression)

                # parse thought from the response
                thought = self.parse_thought(llm_response)

   				...

                # Update prompt with the first response
                print_str += f"""{llm_response}\n{self.stopword}\n"""
                prompt_addition = (
                    f"""{llm_response}\n{self.stopword}{self.planner_config.llm.eot_tag}"""
                )
                self.curr_prompt += prompt_addition
                self.trace += prompt_addition

   				...

                # Parse high level action directives from llm response
                **high_level_actions = self.actions_parser(
                    self.agents, llm_response, self.params
                )**

                # Get low level actions and/or responses
                **low_level_actions, responses = self.process_high_level_actions(
                    high_level_actions, observations
                )**

                # Store last executed high level action
                **self.last_high_level_actions = high_level_actions**

            else: # 아닌 경우는 계속 low_level_actions만 수행
                planner_info["replanned"] = {agent.uid: False for agent in self.agents}
                # Set thought to None
                thought = None

                # Get low level actions and/or responses using last high level actions
                **low_level_actions, responses = self.process_high_level_actions(
                    self.last_high_level_actions, observations
                )**

   				...

   	    def get_agent_collisions(self) -> Dict[int, bool]:
            # Check if the agents are colliding.

   	    def format_response(
            self, response: str, end_expression: Union[str, List[str]]
        ) -> str:
            # Format the LLM response by trimming it up to the first appearance of end_expression.

   	    def parse_thought(self, input_string: str) -> str:
            # Extract thought from the LLM response.

   	    def check_if_agent_done(self, llm_response: str) -> bool:
            # Check if the agent is done based on the LLM response.
    ```

3. Agent

    ```python
    class Agent:
        """
        This class represents an agent, which decides which action should be called at every time step. The agent has access to tools, which will convert high level actions into low-level control commands.
        """

        def __init__(
            self,
            uid: int,
            agent_conf: DictConfig,
            env_interface: Optional["EnvironmentInterface"] = None,
        ) -> None:
            # Initialize Agent

   		...

        def __init_tools(self) -> Dict[str, "Tool"]:
            # Declare as set to ensure uniqueness
            tools = {}

            for tool_category in self.agent_conf.tools:
                for tool_name in self.agent_conf.tools[tool_category]:
                    print(f'processing tool: "{tool_name}"')
                    tool_config = self.agent_conf.tools[tool_category][tool_name]
                    tool = instantiate(tool_config)
                    tool.agent_uid = self.uid

                    **# Motor skills require access to the environment and the device
                    if "motor_skills" in tool_category:
                        tool.set_environment(self.env_interface)
                        tool.to(self.env_interface.device)

                    # Perception requires access to the environment
                    if "perception" in tool_category:
                        tool.set_environment(self.env_interface)**

        ...

        def pass_llm_to_tools(self, llm: "BaseLLM"):
            """
            This method passes a given instance of LLM into the agent tools.
            Some tools require LLMs for their operation. However, maintaining copies of LLMs in memory
            is expensive, so we need to share the instance of planner LLM across the tools.
            :param llm: The llm that drives this agent.
            """
            for tool in self.tools.values():
                if hasattr(tool, "llm"):
                    **tool.set_llm(llm)  # type: ignore**

            return

        def get_last_state_description(self) -> str:
            """
            Obtain the last tool that the agent used.
            """
            if self.last_used_tool == None:
                return "Idle"

            return self.last_used_tool.get_state_description()  # type: ignore

        def process_high_level_action(
            self, action: str, action_input: str, observations: Dict[str, "torch.Tensor"]
        ) -> tuple["torch.Tensor", str]:
            """
            This method will consume high level actions to generate
            either a text response or a low level action.
            For every empty text response, this method should return a non empty low-level action.
            :param action: The name of the high level action (e.g. Pick, Place)
            :param action_input: The argument of the action
            :param observations: current agent observations
            """
            # Fetch tool corresponding to the action
            try:
                tool = self.get_tool_from_name(action)
            except ValueError:
                return None, f'Tool "{action}" not found'

            # Process the high level action
            if self._dry_run and not isinstance(tool, PerceptionTool):
                return None, f"{action} was a success"

            **low_level_action, response = tool.process_high_level_action(
                action_input, observations
            )**

            # Set last used tool
            self.last_used_tool = tool

            return low_level_action, response
    ```

4. WorldGraph (DynamicWorldGraph)

    ```python
    class DynamicWorldGraph(WorldGraph):
        """
        This derived class collects all methods specific to world-graph created and maintained based on observations instead of privileged sim data.
        """

        def __init__(
            self,
            max_neighbors_for_room_assignment: int = 5,
            num_closest_entities_for_entity_matching: int = 5,
            *args,
            **kwargs,
        ):

        def create_cg_edges(
            self,
            cg_dict_list: Optional[dict] = None,
            include_objects: bool = False,
            verbose: bool = False,
        ):
            """
            This method populates the graph from the dict output of CG. Creates a graph to store different entities in the world and their relations to one another. 직접 보시는게 좋을 듯 함
            """

   	    def get_object_from_obs(
            self,
            detector_frame: dict,
            object_id: int,
            uid: int,
            verbose: bool = False,
            object_state_dict: Optional[dict] = None,
        ) -> Optional[Object]:
            """
            Given the processed observation, extract the object's centroid and convert to a
            node
            NOTE: We use Sim information to populate locations for all objects detected by
            Human. Needs to be refactored post bug-fix in KinematicHumanoid class
            @zephirefaith @xavipuig
            """

            ...

            # data / info from sensor being processed ...

            ...

            # add this object to the graph
            **new_object_node = Object(
                f"{len(self._entity_names)+1}_{obj_id_to_category_mapping[object_id]}",
                {
                    "type": obj_id_to_category_mapping[object_id],
                    "translation": object_centroid,
                    "camera_pose_of_view": detector_frame["camera_pose"],
                },
            )**
            # store sim handle for this object; this information is only used to pass
            # to our skills when needed for kinematics simulation. Not used for any privileged perception tasks
            new_object_node.sim_handle = object_handle
            if object_state_dict is not None:
                for state_name, object_state_values in object_state_dict.items():
                    if object_handle in object_state_values:
                        new_object_node.set_state(
                            {state_name: object_state_values[object_handle]}
                        )

            return new_object_node


        def update_non_privileged_graph_with_detected_objects(
            self,
            frame_desc: Dict[int, Dict[str, Any]],
            object_state_dict: dict = None,
            verbose: bool = False,
        ):
            """
            ONLY FOR NON-PRIVILEGED GRAPH SETTING
            This method updates the graph based on the processed observations
            """

            # create masked point-clouds per object and then extract centroid
            # as a proxy for object's location
            # NOTE: using bboxes may also include non-object points to contribute
            # to the object's position...we can fix this with nano-SAM or using
            # analytical approaches to prune object PCD
            for uid, detector_frame in frame_desc.items():
                if detector_frame["object_category_mapping"]:
                    obj_id_to_category_mapping = detector_frame["object_category_mapping"]
                    detector_frame["object_handle_mapping"]  # for sensing states
                    for object_id in detector_frame["object_masks"]:
                        if not self._is_object(obj_id_to_category_mapping[object_id]):
                            continue

                        new_object_node = self.get_object_from_obs(
                            detector_frame,
                            object_id,
                            uid,
                            verbose,
                            object_state_dict=object_state_dict,
                        )

                        if new_object_node is None:
                            continue

                        new_object_node.properties["time_of_update"] = time.time()

                        # add an edge to the closest room to this object
                        # get top N closest objects (N defined by self.max_neighbors_for_room_matching)
                        closest_objects = self.get_closest_entities(
                            self.MAX_NEIGHBORS_FOR_ROOM_ASSIGNMENT,
                            object_node=new_object_node,
                            include_furniture=False,
                            include_rooms=False,
                        )

                        ...

                        # only add this object is it is not being held by an agent
                        # or if it is not already in the world-graph
                        skip_adding_object = redundant_object | held_object

                        if skip_adding_object:
                            # update the matching object's translation and states
                            continue

                        **self.add_node(new_object_node)**
                        self._entity_names.append(new_object_node.name)
                        self._logger.info(f"Added new object to CG: {new_object_node}")
                        **reference_furniture, relation = self._cg_check_for_relation(
                            new_object_node
                        )
                        if reference_furniture is not None and relation is not None:
                            self.add_edge(
                                reference_furniture,
                                new_object_node,
                                relation,
                                flip_edge(relation),**
                            )
                        else:
                            # if **not redundant and not belonging to a furniture**
                            # then find the room this object should belong to
                            # find most common room among these objects
                            # TODO: get closest objects but only consider those visible to agent
                            room_counts: Dict[Union[Object, Furniture], int] = {}

                            for obj in closest_objects:
                                for room in self.get_neighbors_of_type(obj, Room):
                                    if verbose:
                                        self._logger.info(
                                            f"Adding {new_object_node.name} --> Closest object: {obj.name} is in room: {room.name}"
                                        )
                                    if room in room_counts:
                                        room_counts[room] += 1
                                    else:
                                        room_counts[room] = 1
                                    # only use the first Room neighbor, i.e. closest room node
                                    break

                            **if room_counts:
                                closest_room = max(room_counts, key=room_counts.get)
                                self.add_edge(
                                    new_object_node,
                                    closest_room,
                                    "in",
                                    opposite_label="contains",
                                )**

   	    def update_by_action(
            self,
            agent_uid: int,
            high_level_action: Tuple[str, str, Optional[str]],
            action_response: str,
            verbose: bool = False,
        ):
            """
            Deterministically updates the world-graph based on last-action taken by agent_{agent_uid} based on the result of that action.
            Only updates the graph if the action was successful. Applicable only when one wants to change agent_{agent_uid}'s graph
            based on agent_{agent_uid}'s actions.

            Please look at update_by_other_agent_action or update_non_privileged_graph_by_other_agent_action for updating self graph based on another agent's actions.
            """

        def update_non_privileged_graph_by_action(
            self,
            agent_uid: int,
            high_level_action: Tuple[str, str, Optional[str]],
            action_response: str,
            verbose: bool = False,
            drop_placed_object_flag: bool = True,
        ):
            """
            ONLY FOR USE WITH NON-PRIVILEGED GRAPH

            Deterministically updates the world-graph based on last-action taken by agent_{agent_uid} based on the result of that action.
            Only updates the graph if the action was successful. Applicable only when one wants to change agent_{agent_uid}'s graph
            based on agent_{agent_uid}'s actions. If drop_placed_object_flag is True then whenever an object is placed it is simply deleter from the graph instead of being read to the receptacle.
            This method is different from update_by_action as it expects non-privileged entities as input and not GT sim entities.

            Please look at update_by_other_agent_action or update_non_privileged_graph_by_other_agent_action for updating self graph based on another agent's actions.
            """

        def update_non_privileged_graph_by_other_agent_action(
            self,
            other_agent_uid: int,
            high_level_action_and_args: Tuple[str, str, Optional[str]],
            action_results: str,
            verbose: bool = False,
            drop_placed_object_flag: bool = True,
        ):

        def update_by_other_agent_action(
            self,
            other_agent_uid: int,
            high_level_action_and_args: Tuple[str, str, Optional[str]],
            action_results: str,
            use_semantic_similarity: bool = False,
            verbose: bool = False,
        ):

        # 등등 존재
   ```

5. Perception

    ```python
    class PerceptionObs(PerceptionSim):
        """
        This class uses only the simulated panoptic sensors to detect objects and then ground there location based on depth images being streamed by the agents. Note that no other privileged information about the state of the world is used to enhance object location or inter-object relations. We use previously detected objects and furniture through CG to ground properties of newer objects detected through panoptic sensors.
        """

        def __init__(self, sim, metadata_dict: Dict[str, str], *args, **kwargs):
            super().__init__(sim, metadata_dict=metadata_dict, detectors=["gt_panoptic"])

        def preprocess_obs_for_non_privileged_graph_update(
            self, sim, obs: Dict[str, Any], single_agent_mode: bool = False
        ) -> List[Dict[str, Any]]:
            """
            ONLY FOR NON-PRIVILEGED GRAPH SETTING
            Creates a list of observations for each agent in the scene. Each observation  contains the RGB, depth, masks, camera intrinsics and camera pose for the agent. 
            **이처럼 processing을 habitat_llm 안에서 수행**
            """

            ...

        def get_sim_handle_and_key_from_panoptic_image(
            self, obs: np.ndarray
        ) -> Dict[int, str]:
            """
            This method uses the instance segmentation output to create a list of handles of all objects present in given agent's FOV
            """
   			
            ...


        def get_object_detections_for_non_privileged_graph_update(
            self, input_obs: List[Dict[str, Any]]
        ) -> Dict[int, Dict[str, Any]]:
            """
            ONLY FOR NON-PRIVILEGED GRAPH SETTING
            Use the panoptic sensor to detect objects in the agent's FOV

            NOTE: We calculate location for objects seen by robot using RGB-D images and camera intrinsics + extrinsics. For Human we use sim information as there is a known bug in using the RGB-D images for humans.
            """
            
            ...

    ```

## Code Organization

Below are the details of various important directories and classes.

- **habitat-llm**
  - **Agent** : Represents the robot or human. An agent can act in the environment.
  - **Tools** : Represents abstractions which enable the agent to perceive or act in the environment.
  - **Planner** : Represents centralized and decentralized planners.
  - **LLM** : Contains abstractions to represent Llama and GPT APIs.
  - **WorldGraph** : Contains a hierarchical world graph representing rooms, furniture, objects.
  - **Perception** : Contains a simulated perception pipeline which sends local detections to the world model.
  - **Examples**: Contains demos and evaluation programs to show or analyze the performance of planners.
  - **EvaluationRunner**: Represents an abstraction for running the planners.
  - **Conf** : Contains hydra config files for all classes. You can set the experiments by changing these configures. (ex. agent, baselines, evaluation, ...) You can check the example configure with **"./example_conf.yaml"**
  - **Utils** : Contains various utility methods required throughout the codebase.
  - **Tests** : Contains unit tests.
- **scripts**
  - **hitl_analysis** : Contains scripts to analyze and replay human-in-the-loop traces.
  - **prediviz** : Contains visualization and annotation tools for PARTNR tasks.

## Example and Use Cases

1. 사용자 친화 예제 몇개 더 보여주기

   추후 추가 예정

2. 사용자 정의 agent 개발

   추후 추가 예정

## Others

partnr-planner 원본 참고하시면 더 많은 정보 가능
