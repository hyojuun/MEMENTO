# Configuration Files for Habitat LLM

This directory contains configuration files used for various components of the Habitat LLM framework. These configurations are essential for setting up agents, planners, evaluations, and other components of the system.


## Habitat LLM Configuration Settings

This is the overall structure of the Habitat LLM configuration settings. You can check the example configure with **"EmbodiedAgent/example_conf.yaml"**.

- **habitat_baselines**: Settings for running habitat baselines, which include configurations for training algorithms, evaluation modes, and environment management. This allows users to define how agents are trained and evaluated within the Habitat framework, ensuring efficient learning and performance monitoring.
- **habitat**: This allows you to configure the main settings for the habitat-lab simulator, for example tasks, datasets, agents with sensors, environments, and navigation actions, etc... For a comprehensive list of configuration keys available in habitat-lab, please check the documentation at https://github.com/facebookresearch/habitat-lab/blob/main/habitat-lab/habitat/config/CONFIG_KEYS.md.
- **evaluation**: Settings used for the evaluation, especially for agent settings. What tools for agents and how to set the agents' planner.
- **world_model**: Graph, partial observabilities..
- **trajectory**: Options to deal with the trajectories.
- **agents**: Whole settings about the agents in the simulator.
- **WANDB**: Setting of Weights & Biases (WANDB)
- **others**: Other settings like agent_asymmetry, device, instruction mode, env, num_runs_per_episode, num_proc, dry_run, robot_agent_uid, human_agent_uid, paths.


## Directory Structure

You can build those configure with these separated hydra yaml files.

- **agent/**: Contains configuration files for different types of agents. You can set perception and motor skills with this .yaml files.
- **baselines/**: Contains configuration files for baseline experiments. This configure represents the highest level, set what LLM, agent as a planner, how many agents, etc...
- **benchmark_gen/**: Contains configuration files for evaluation settings. 
- **evaluation/**: Contains configuration files for evaluation settings. You can set the evaluation methods like save_video, output_dir, more detailed settings for evaluation, uid for agents, etc...
- **examples/**: Contains the examples show how to set the Hydra yaml settings with other separated configures.
- **finetuning/**: Contains configuration files for finetuning.
- **habitat_conf/**: Contains configuration files for habitat simulator, which is simulator, gym, environments etc...
    - **habitat_conf/dataset**: Configure about what dataset are you going to use.
    - **habitat_conf/habitat_agent**: Configure about what the agents, especially what sensor is used.
    - **habitat_conf/task**: Contains configuration files for the tasks. You can explicitly set each agents' objectives, sensors, measurements and actions.
- **hydra/**: Configure about hydra settings.
- **instruct/**: Configure about the prompts and some signs, parsers which are injected to planners.
- **llm/**: Configure about the llm.
- **logging/**: Configure (Option) about saving.
- **map/**: Configure about the maps.
- **planner/**: Configure about the planner like planning mode (ex. CoT), using RAG, replanning_thresholds, etc...
- **rlm/**: Maybe it is related with resource.
- **tools/**: Configure about the motor_skills and perception, etc... that agent can use.
- **training/**: Configure about the base training methods.
- **trajectory/**: Configure about the trajectory loggers.
- **wandb_conf/**: Configure about the wandb configures.
- **world_model/**: Configure about the world models (concept_graph, gt_graph).

We can set each folder's yaml at examples or baselines to make a habitat-llm config.


## Simple Code Logic

Based on the Hydra configures we gave, register sensors and initialize the lab interface with *EnvironmentInterface*. After this, construct agents with EvaluationRunner, and run the agent with the episodes.
And you can check the example for running at examples/planner_demo.py. 

---

## Agent Configurations

### oracle_rearrange_agent_motortoolsonly.yaml
This configuration defines an oracle agent that can perform rearrangement tasks using motor tools only. It includes parameters for the agent's capabilities and behavior.

### oracle_nav_agent.yaml
This configuration defines an oracle navigation agent. It specifies how the agent should navigate through the environment, including any specific navigation strategies or tools it can use.

### nn_rearrange_agent.yaml
This configuration defines a neural network-based rearrangement agent. It includes settings for the neural network model, training parameters, and any specific behaviors the agent should exhibit.

### nn_rearrange_agent_motortoolsonly.yaml
This configuration defines a neural network-based rearrangement agent that uses motor tools only. Similar to the previous configuration, it specifies the neural network model and its capabilities.

## Baseline Configurations

The baseline configurations are designed to provide a starting point for experiments. They include predefined settings for various tasks and can be modified to suit specific needs.

## Evaluation Configurations

These configurations define how the evaluation of agents and tasks should be conducted. They include metrics to be used, evaluation strategies, and any specific settings required for the evaluation process.

## Usage

To use these configuration files, you can specify them in your command line when running experiments or simulations. For example:
```bash
python -m habitat_llm.examples.planner_demo --config-name baselines/single_agent_zero_shot_react_summary.yaml
```

Make sure to adjust the paths and parameters according to your specific setup and requirements.

## Customizing Configurations

You can customize these configuration files to fit your specific needs. This may include changing parameters, adding new agents, or modifying evaluation metrics. Be sure to document any changes you make for future reference. You can refer EmbodiedAgent/docs/extending.md

## Conclusion

The configuration files in this directory are crucial for the operation of the Habitat LLM framework. Understanding and customizing these files will allow you to effectively utilize the framework for your research and development needs.

