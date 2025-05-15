# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import attr
import numpy as np

from habitat.core.registry import registry
from habitat.core.utils import DatasetFloatJSONEncoder
from habitat.datasets.utils import check_and_gen_physics_config

from habitat_llm.agent.env.dataset import (
    CollaborationEpisode,
    CollaborationDatasetV0
)
import habitat_llm.agent.env.evaluation.evaluation_functions as evaluation_functions
from habitat_llm.agent.env.evaluation.evaluation_functions import (
    EvaluationConstraint,
    EvaluationProposition,
    EvaluationPropositionDependency,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig






@attr.s(auto_attribs=True, kw_only=True)
class MemoryManagementEpisode(CollaborationEpisode):
    """Specifies additional instruction and evaluation data for a particular instance of a memory_management task.

    For a definition of inherited keys, see CollaborationEpisode.
    """

    metadata: Dict[str, Any] = attr.ib(factory=dict)
    original_data_info: Dict[str, Any] = attr.ib(factory=dict)
    


@registry.register_dataset(name="MemoryManagementDataset-v0")
class MemoryManagementDatasetV0(CollaborationDatasetV0):
    episodes: List[MemoryManagementEpisode]

    def __init__(
        self,
        config: Optional["DictConfig"] = None,
        episodes: List[MemoryManagementEpisode] = None,
    ) -> None:
        self.config = config
        if episodes is not None:
            # If the episodes are given, init the dataset with these episodes.
            # We do this so that we can load a partial dataset.
            self.episodes = episodes
        else:
            # Otherwise, init the dataset with the episode specified in config
            if config and not self.check_config_paths_exist(config):
                data_p = config.data_path.format(split=config.split)
                scenes_p = config.scenes_dir
                raise ValueError(
                    f"Collaboration task assets are not downloaded locally. Either {data_p} or {scenes_p} do not exist."
                )

            check_and_gen_physics_config()
            super(CollaborationDatasetV0, self).__init__(config)

    def apply_scene_dir_prefix(
        self, episode: MemoryManagementEpisode, scenes_dir: Optional[str] = None
    ) -> MemoryManagementEpisode:
        """Overrides the scene directory to `scene_dataset_config` if provided."""
        if not scenes_dir:
            return episode

        episode.scene_dataset_config = os.path.join(
            scenes_dir, os.path.basename(episode.scene_dataset_config)
        )
        return episode

    def to_json(self) -> str:
        """Serializes the current dataset into a string JSON representation."""
        tmp_cfg = self.config
        self.config = None
        result = DatasetFloatJSONEncoder().encode(self)
        self.config = tmp_cfg
        return result

    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:
        """Deserializes a dataset from a string JSON representation."""
        deserialized = json.loads(json_str)

        for episode in deserialized["episodes"]:
            memory_management_ep = MemoryManagementEpisode(**episode)

            for i, prop in enumerate(memory_management_ep.evaluation_propositions):
                memory_management_ep.evaluation_propositions[i] = EvaluationProposition(
                    **prop
                )  # type: ignore

            for i, dep in enumerate(
                memory_management_ep.evaluation_proposition_dependencies
            ):
                memory_management_ep.evaluation_proposition_dependencies[
                    i
                ] = EvaluationPropositionDependency(
                    **dep
                )  # type: ignore

            for i, constraint in enumerate(memory_management_ep.evaluation_constraints):
                constraint_cls = getattr(evaluation_functions, constraint["type"])  # type: ignore
                memory_management_ep.evaluation_constraints[i] = constraint_cls(
                    **constraint["args"]  # type: ignore
                )

            memory_management_ep = self.apply_scene_dir_prefix(memory_management_ep, scenes_dir)
            self.episodes.append(memory_management_ep)

    def from_binary(
        self, data_dict: Dict[str, Any], scenes_dir: Optional[str] = None
    ) -> None:
        """Load the dataset from a pickle compatible Dict."""
        all_T = data_dict["all_transforms"]
        idx_to_name = data_dict["idx_to_name"]
        for ep in data_dict["all_eps"]:
            ep["rigid_objs"] = [
                [idx_to_name[ni], all_T[ti]] for ni, ti in ep["rigid_objs"]
            ]
            ep["ao_states"] = {idx_to_name[ni]: v for ni, v in ep["ao_states"].items()}
            ep["name_to_receptacle"] = {
                idx_to_name[k]: idx_to_name[v] for k, v in ep["name_to_receptacle"]
            }

            new_markers = []
            for name, mtype, offset, link, obj in ep["markers"]:
                new_markers.append(
                    {
                        "name": idx_to_name[name],
                        "type": idx_to_name[mtype],
                        "params": {
                            "offset": offset,
                            "link": idx_to_name[link],
                            "object": idx_to_name[obj],
                        },
                    }
                )
            ep["markers"] = new_markers

            memory_management_ep = MemoryManagementEpisode(**ep)
            memory_management_ep = self.apply_scene_dir_prefix(memory_management_ep, scenes_dir)
            self.episodes.append(memory_management_ep)
