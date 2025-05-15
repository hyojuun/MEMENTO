#!/usr/bin/env python3

"""
This module contains DescribeObjectTool, a PerceptionTool, used by tool-based LLM to
retrieve a brief descriptive or semantic meaning of a given object or furniture name.
"""

from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from src.agent.env.environment_interface import EnvironmentInterface

import numpy as np
import pandas as pd
import re

from habitat_llm.utils.grammar import FREE_TEXT
from habitat_llm.world_model import Furniture, Human, Receptacle, Room, SpotRobot

from src.tools import PerceptionTool


class DescribeObjectTool(PerceptionTool):
    """
    PerceptionTool used by tool-based LLM planner to get the exact identifier for an
    object node given natural language description of the object.
    """

    def __init__(self, skill_config):
        super().__init__(skill_config.name)
        self.env_interface = None
        self.skill_config = skill_config
        self.category = skill_config.category
        self.caption_df = None
        self.read_caption_file()
        
    def read_caption_file(self):
        self.caption_df = pd.read_csv(self.skill_config.caption_file_path)
        
    def get_object_handle(self, object_name: str) -> Tuple[str, bool]:
        if object_name in self.env_interface.perception.sim_name_to_handle:
            return self.env_interface.perception.sim_name_to_handle[object_name], True
        else:
            return f"Object '{object_name}' not found in the environment.", False
        
    def parse_object_handle(self, object_handle: str) -> str:
        # Return the substring before '_:'
        return object_handle.split('_:')[0]

    def set_environment(self, env_interface: "EnvironmentInterface") -> None:
        """
        Sets the tool's environment_interface var
        :param env: EnvironmentInterface instance associated with episode
        """
        self.env_interface = env_interface
        
    @property
    def description(self) -> str:
        """
        property to return the description of this tool as provided in configuration
        :return: tool description
        """
        return self.skill_config.description
    
    def process_high_level_action(
        self, input_query: str, observations: dict
    ) -> Tuple[None, str]:
        """
        Main entry-point, takes the natural language object query as input and the latest
        observation. Returns the exact name of the object node matching query.

        :param input_query: Natural language description of object of interest
        :param observations: Dict of agent's observations

        :return: Tuple[None, str], where the 2nd element is the exact name of the node matching
        input-query, or a message explaining such object was not found.
        """
        super().process_high_level_action(input_query, observations)
        
        # Get the object handle
        object_handle, found = self.get_object_handle(input_query)
        if not found:
            return None, object_handle
        
        # Parse the object handle
        parsed_object_handle = self.parse_object_handle(object_handle)
        # logging.info(f"Parsed object handle: {parsed_object_handle}")
        
        # Get the caption for the object
        caption_row = self.caption_df[self.caption_df['id'] == parsed_object_handle]
        if caption_row.empty:
            answer = f"No description found for '{input_query}'."
        else:
            answer = f"The description of the object '{input_query}' is:\n{caption_row['caption'].values[0]}"
        

        # Handle the edge case where answer is empty or only spaces
        if answer == "" or answer.isspace():
            answer = (
                f"Could not find any object in world for the query '{input_query}'."
            )

        return None, answer

    @property
    def argument_types(self) -> List[str]:
        """
        Returns the types of arguments required for the FindObjectTool.

        :return: List of argument types.
        """
        return [FREE_TEXT]
