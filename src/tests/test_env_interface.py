# TODO: test env interface
# 1. environment interface 초기화
# 2. env, sim, device 등 configuration 설정 잘 되는지
# 3. get_observations()
# 4. parse_observations()
# 5. reset_environment()
# 6. step() 호출 시 정상 동작
# 7. action 입력 시 반환 여부

import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# from agent.env.environment_interface import EnvironmentInterface
import hydra
from habitat_llm.utils.core import fix_config, setup_config

from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
    remove_visual_sensors,
)

from habitat_llm.agent.env.dataset import CollaborationDatasetV0


@hydra.main(config_path="../conf")
def main(config):
    fix_config(config)
    # Setup a seed
    seed = 47668090
    # Setup config
    config = setup_config(config, seed)
    dataset = CollaborationDatasetV0(config.habitat.dataset)
    
    if config.env == "habitat":
        print("habitat environment")
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

        # config.world_model.type = "concept_graph"
        
        # Initialize the environment interface for the agent
        env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=False)

        try:
            env_interface.initialize_perception_and_world_graph()
        except Exception:
            print("Error initializing the environment")
        
        # print(dir(env_interface.world_graph))
        print(env_interface.world_graph)
        print(env_interface.world_graph[0].world_model_type)
        print(len(env_interface.world_graph[0].get_all_furnitures()))
        print(len(env_interface.world_graph[0].get_all_objects()))
        print(len(env_interface.world_graph[0].get_all_receptacles()))
        print(len(env_interface.world_graph[0].get_all_rooms()))
        print(env_interface.world_graph[1].world_model_type)
        print(len(env_interface.world_graph[1].get_all_furnitures()))
        print(len(env_interface.world_graph[1].get_all_objects()))
        print(len(env_interface.world_graph[1].get_all_receptacles()))
        print(len(env_interface.world_graph[1].get_all_rooms()))
        
if __name__ == "__main__":
    main()