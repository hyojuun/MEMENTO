import json
import os
import copy
import random
import argparse
from collections import defaultdict, Counter

# Default file paths
DEFAULT_INPUT_FILE = 'data/datasets/single_memory_utilization_stage.json'
DEFAULT_OUTPUT_FILE = 'data/datasets/joint_memory_utilization_stage.json'

# Define combination types
COMBINATION_TYPES = [
    "Object Semantics + Object Semantics",
    "Object Semantics + User Pattern",
    "User Pattern + User Pattern"
]

def load_dataset(file_path):
    """Load dataset"""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_dataset(dataset, file_path):
    """Save dataset"""
    with open(file_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"Dataset saved to {file_path}.")

def group_episodes_by_scene_and_type(episodes):
    """Group episodes by scene_id and type"""
    scene_episodes = defaultdict(lambda: defaultdict(list))
    
    for episode in episodes:
        scene_id = episode['scene_id']
        
        # Check episode type (from metadata)
        if 'metadata' in episode and 'episode_type' in episode['metadata']:
            episode_type = episode['metadata']['episode_type']
        else:
            # Assign 'object_semantics' as default
            episode_type = 'object_semantics'
        
        scene_episodes[scene_id][episode_type].append(episode)
    
    return scene_episodes

def generate_combined_instruction(ep1, ep2, combination_type):
    """Combine instructions from two episodes"""
    instr1 = ep1['instruction']
    instr2 = ep2['instruction']
    
    # Connector phrase based on combination type
    if combination_type == "Object Semantics + Object Semantics":
        connector = "Then, "
    elif combination_type == "Object Semantics + User Pattern":
        connector = "After that, "
    elif combination_type == "User Pattern + User Pattern":
        connector = "Additionally, "
    
    # Convert first letter of second instruction to lowercase if it's uppercase
    if instr2 and instr2[0].isupper():
        instr2 = instr2[0].lower() + instr2[1:]
    
    combined = f"{instr1} {connector}{instr2}" if instr2 else instr1
    
    return combined

def check_receptacle_overlaps(ep1, ep2):
    """Check for receptacle overlaps between two episodes"""
    # Extract name_to_receptacle from each episode
    receptacles1 = ep1.get('name_to_receptacle', {})
    receptacles2 = ep2.get('name_to_receptacle', {})
    
    # Find common receptacles
    common_receptacles = set(receptacles1.keys()) & set(receptacles2.keys())
    
    return {
        "has_overlaps": len(common_receptacles) > 0,
        "common_receptacles": list(common_receptacles),
        "overlap_count": len(common_receptacles)
    }

def combine_evaluation_propositions(ep1, ep2):
    """Combine evaluation propositions from two episodes"""
    props1 = ep1['evaluation_propositions']
    props2 = ep2['evaluation_propositions']
    
    # Combined propositions
    combined_props = copy.deepcopy(props1)
    combined_props.extend(copy.deepcopy(props2))
    
    return combined_props

def combine_evaluation_constraints(ep1, ep2):
    """Combine evaluation constraints from two episodes"""
    constraints1 = ep1['evaluation_constraints']
    constraints2 = ep2['evaluation_constraints']
    
    # Number of propositions in the first episode
    n_props1 = len(ep1['evaluation_propositions'])
    
    # Combined constraints list
    combined_constraints = []
    
    # Combine temporal constraints (TemporalConstraint)
    temporal_constraint_found = False
    
    # Find TemporalConstraint in both episodes
    # First, check the first episode
    temporal_constraint1 = None
    for c in constraints1:
        if c['type'] == 'TemporalConstraint':
            temporal_constraint1 = copy.deepcopy(c)
            temporal_constraint_found = True
            break
    
    # Then check the second episode
    temporal_constraint2 = None
    for c in constraints2:
        if c['type'] == 'TemporalConstraint':
            temporal_constraint2 = copy.deepcopy(c)
            temporal_constraint_found = True
            break
    
    # Process if at least one temporal constraint exists
    if temporal_constraint_found:
        resulting_temporal_constraint = None
        
        # If first episode has temporal constraint, use it as base
        if temporal_constraint1:
            resulting_temporal_constraint = temporal_constraint1
            edges1 = resulting_temporal_constraint['args'].get('dag_edges', [])
        else:
            # If first episode has no temporal constraint, create a new one
            resulting_temporal_constraint = {
                'type': 'TemporalConstraint',
                'args': {
                    'dag_edges': [],
                    'n_propositions': n_props1
                }
            }
            edges1 = []
        
        # Process temporal constraint from second episode
        if temporal_constraint2:
            # Adjust DAG edge indices from second episode
            adjusted_dag_edges2 = [
                [edge[0] + n_props1, edge[1] + n_props1] 
                for edge in temporal_constraint2['args'].get('dag_edges', [])
            ]
        else:
            adjusted_dag_edges2 = []
        
        # Add new edges connecting all propositions from first episode to the first proposition of second episode
        new_edges = []
        for i in range(n_props1):
            # Connect to first proposition of second episode (index n_props1)
            new_edges.append([i, n_props1])
        
        # Combine all edges
        combined_dag_edges = edges1 + adjusted_dag_edges2 + new_edges
        
        # Update the temporal constraint
        resulting_temporal_constraint['args']['dag_edges'] = combined_dag_edges
        resulting_temporal_constraint['args']['n_propositions'] = n_props1 + len(ep2['evaluation_propositions'])
        
        combined_constraints.append(resulting_temporal_constraint)
    
    # Combine TerminalSatisfactionConstraint
    terminal_constraint = None
    for c in constraints1:
        if c['type'] == 'TerminalSatisfactionConstraint':
            terminal_constraint = copy.deepcopy(c)
            break
    
    if terminal_constraint:
        # Find terminal constraint in second episode
        terminal2 = None
        for c in constraints2:
            if c['type'] == 'TerminalSatisfactionConstraint':
                terminal2 = c
                break
        
        # Combine terminal proposition indices from both episodes
        if terminal2:
            # Adjust indices from second episode
            adjusted_indices2 = [idx + n_props1 for idx in terminal2['args'].get('proposition_indices', [])]
            
            # Combine all indices
            combined_indices = terminal_constraint['args'].get('proposition_indices', []) + adjusted_indices2
            
            # Create new terminal constraint
            terminal_constraint['args']['proposition_indices'] = combined_indices
        
        combined_constraints.append(terminal_constraint)
    
    return combined_constraints

def detect_object_overlaps(ep1, ep2):
    """Detect object overlaps between two episodes"""
    metadata1 = ep1.get('metadata', {})
    metadata2 = ep2.get('metadata', {})
    
    # Extract target objects
    target_objects1 = metadata1.get('target_objects', [])
    target_objects2 = metadata2.get('target_objects', [])
    
    # Extract distractor objects
    distractors1 = metadata1.get('distractors', [])
    distractors2 = metadata2.get('distractors', [])
    
    # Extract object IDs (if object is a dictionary, use 'id' key, otherwise use the object itself as ID)
    def extract_object_ids(objects):
        ids = []
        for obj in objects:
            if isinstance(obj, dict) and 'id' in obj:
                ids.append(obj['id'])
            elif isinstance(obj, dict) and 'object_id' in obj:
                ids.append(obj['object_id'])
            else:
                ids.append(str(obj))
        return set(ids)
    
    # Extract object categories
    def extract_object_categories(objects):
        categories = []
        for obj in objects:
            if isinstance(obj, dict) and 'category' in obj:
                categories.append(obj['category'])
            elif isinstance(obj, dict) and 'object_category' in obj:
                categories.append(obj['object_category'])
            # Process objects from initial_state in info field
            elif isinstance(obj, dict) and 'object_classes' in obj and isinstance(obj['object_classes'], list) and len(obj['object_classes']) > 0:
                categories.append(obj['object_classes'][0])
        return set(categories)
    
    # Extract IDs from each list
    target_ids1 = extract_object_ids(target_objects1)
    target_ids2 = extract_object_ids(target_objects2)
    distractor_ids1 = extract_object_ids(distractors1)
    distractor_ids2 = extract_object_ids(distractors2)
    
    # Extract categories from each list
    target_categories1 = extract_object_categories(target_objects1)
    target_categories2 = extract_object_categories(target_objects2)
    
    # Detect overlaps based on IDs
    target_target_overlap = target_ids1.intersection(target_ids2)
    distractor_distractor_overlap = distractor_ids1.intersection(distractor_ids2)
    
    # Detect cross-overlaps between targets and distractors
    target1_distractor2_overlap = target_ids1.intersection(distractor_ids2)
    target2_distractor1_overlap = target_ids2.intersection(distractor_ids1)
    
    # Detect category-based overlaps (between target objects)
    target_category_overlap = target_categories1.intersection(target_categories2)
    
    # Construct results
    overlaps = {
        'target_target': list(target_target_overlap),
        'distractor_distractor': list(distractor_distractor_overlap),
        'target1_distractor2': list(target1_distractor2_overlap),
        'target2_distractor1': list(target2_distractor1_overlap),
        'target_category_overlap': list(target_category_overlap),
        'counts': {
            'target_target': len(target_target_overlap),
            'distractor_distractor': len(distractor_distractor_overlap),
            'target1_distractor2': len(target1_distractor2_overlap),
            'target2_distractor1': len(target2_distractor1_overlap),
            'target_category_overlap': len(target_category_overlap),
            'total_overlap': len(target_target_overlap) + len(distractor_distractor_overlap) +
                             len(target1_distractor2_overlap) + len(target2_distractor1_overlap)
        },
        'has_target_category_overlap': len(target_category_overlap) > 0,
        'has_target_distractor_overlap': len(target1_distractor2_overlap) > 0 or len(target2_distractor1_overlap) > 0
    }
    
    return overlaps

def combine_metadata(ep1, ep2, combination_type):
    """Combine metadata from two episodes"""
    metadata1 = ep1.get('metadata', {})
    metadata2 = ep2.get('metadata', {})
    
    # Create empty dictionary for combined metadata
    combined_metadata = {}
    
    # Detect object overlaps
    overlaps = detect_object_overlaps(ep1, ep2)
    
    # Store all metadata fields as lists
    # Create a list of all keys from both episodes
    all_keys = set(list(metadata1.keys()) + list(metadata2.keys()))
    
    # List of fields that need special handling
    special_fields = ['target_objects', 'distractors', 'used_object', 'stage']
    
    # Combine each key in list form
    for key in all_keys:
        # Skip fields that need special handling (process them later)
        if key in special_fields:
            continue
            
        # If key exists in both episodes
        if key in metadata1 and key in metadata2:
            combined_metadata[key] = [metadata1[key], metadata2[key]]
        # If key exists only in the first episode
        elif key in metadata1:
            combined_metadata[key] = [metadata1[key], None]
        # If key exists only in the second episode
        elif key in metadata2:
            combined_metadata[key] = [None, metadata2[key]]
    
    # Process fields that need special handling
    # 1. Keep 'stage' field as a single value (determined by input file)
    if 'stage' in metadata1:
        combined_metadata['stage'] = metadata1['stage']
    elif 'stage' in metadata2:
        combined_metadata['stage'] = metadata2['stage']
    else:
        # Try to extract stage value from file path
        if 'stage1' in DEFAULT_INPUT_FILE:
            combined_metadata['stage'] = 1
        elif 'stage2' in DEFAULT_INPUT_FILE:
            combined_metadata['stage'] = 2
    
    # 2. target_objects and distractors - preserve values from both episodes as lists
    target_objects1 = metadata1.get('target_objects', [])
    target_objects2 = metadata2.get('target_objects', [])
    combined_metadata['target_objects_original'] = [target_objects1, target_objects2]
    # Also maintain combined form as before
    combined_metadata['target_objects'] = target_objects1 + target_objects2
    
    distractors1 = metadata1.get('distractors', [])
    distractors2 = metadata2.get('distractors', [])
    combined_metadata['distractors_original'] = [distractors1, distractors2]
    # Also maintain combined form as before
    combined_metadata['distractors'] = distractors1 + distractors2
    
    # 3. used_object - maintain original processing method while preserving original form
    used_object1 = metadata1.get('used_object', [])
    used_object2 = metadata2.get('used_object', [])
    
    # Convert to lists
    if not isinstance(used_object1, list):
        used_object1 = [used_object1] if used_object1 else []
    if not isinstance(used_object2, list):
        used_object2 = [used_object2] if used_object2 else []
    
    # Preserve original form
    combined_metadata['used_object_original'] = [used_object1, used_object2]
    # Maintain combined form
    combined_metadata['used_object'] = used_object1 + used_object2
    
    # Set episode type (this field is not stored as a list)
    combined_metadata['episode_type'] = 'combined'
    combined_metadata['combination_type'] = combination_type
    
    # Add object overlap information (this field is not stored as a list)
    combined_metadata['object_overlaps'] = overlaps
    
    # Add receptacle overlap information (this field is not stored as a list)
    receptacle_overlaps = check_receptacle_overlaps(ep1, ep2)
    combined_metadata['receptacle_overlaps'] = receptacle_overlaps
    
    return combined_metadata

def merge_info_fields(info1, info2):
    """Merge info fields from two episodes"""
    # Handle case when info field is missing
    if not info1:
        return copy.deepcopy(info2) if info2 else {}
    if not info2:
        return copy.deepcopy(info1)
    
    merged_info = copy.deepcopy(info1)
    
    # Merge additional information from the second episode's info
    for key, value in info2.items():
        if key not in merged_info:
            merged_info[key] = value
        elif key == 'initial_state':
            # Merge initial_state - include initial_state from both episodes
            try:
                # If initial_state is a list
                if isinstance(merged_info[key], list) and isinstance(value, list):
                    # Merge while removing duplicates
                    existing_states = set(str(item) for item in merged_info[key])
                    for item in value:
                        if str(item) not in existing_states:
                            merged_info[key].append(item)
                # If only one is a list
                elif isinstance(merged_info[key], list):
                    if str(value) not in set(str(item) for item in merged_info[key]):
                        merged_info[key].append(value)
                elif isinstance(value, list):
                    temp_list = [merged_info[key]]
                    for item in value:
                        if str(item) != str(merged_info[key]):
                            temp_list.append(item)
                    merged_info[key] = temp_list
                # If neither is a list
                else:
                    if merged_info[key] != value:
                        merged_info[key] = [merged_info[key], value]
            except Exception as e:
                print(f"Error while merging initial_state: {e}")
    
    return merged_info

def merge_rigid_objs(rigid_objs1, rigid_objs2):
    """Merge rigid_objs fields from two episodes"""
    # Check structure of rigid_objs and merge appropriately
    
    # Handle case when rigid_objs is missing or empty
    if not rigid_objs1:
        return rigid_objs2
    if not rigid_objs2:
        return rigid_objs1
    
    # If first item of rigid_objs is not a list or dictionary (basic type)
    if not isinstance(rigid_objs1[0], (dict, list)):
        # Merge two lists while removing duplicates
        return list(set(rigid_objs1 + rigid_objs2))
    
    # If first item of rigid_objs is a list
    if isinstance(rigid_objs1[0], list):
        # Simply concatenate the two lists
        return rigid_objs1 + rigid_objs2
    
    # If first item of rigid_objs is a dictionary
    # Remove duplicates based on object ID
    objects_map = {}
    
    # Process rigid_objs from first episode
    for obj in rigid_objs1:
        if isinstance(obj, dict):
            obj_id = obj.get('object_id')
            if obj_id:
                objects_map[obj_id] = obj
        else:
            # If not a dictionary, use index as ID
            objects_map[f"obj1_{rigid_objs1.index(obj)}"] = obj
    
    # Process rigid_objs from second episode
    for obj in rigid_objs2:
        if isinstance(obj, dict):
            obj_id = obj.get('object_id')
            if obj_id and obj_id not in objects_map:  # Remove duplicates
                objects_map[obj_id] = obj
        else:
            # If not a dictionary, use index as ID
            objects_map[f"obj2_{rigid_objs2.index(obj)}"] = obj
    
    return list(objects_map.values())

def merge_name_to_receptacle(receptacles1, receptacles2):
    """Merge name_to_receptacle fields from two episodes"""
    # Handle missing fields
    if not receptacles1:
        return receptacles2
    if not receptacles2:
        return receptacles1
    
    # Check type of each field
    if not isinstance(receptacles1, dict) or not isinstance(receptacles2, dict):
        print(f"name_to_receptacle field format error: first={type(receptacles1)}, second={type(receptacles2)}")
        # If not a dictionary, return value from first episode
        return receptacles1 if isinstance(receptacles1, dict) else receptacles2 if isinstance(receptacles2, dict) else {}
    
    merged_receptacles = copy.deepcopy(receptacles1)
    
    # Add receptacles from second episode
    for name, value in receptacles2.items():
        if name not in merged_receptacles:
            merged_receptacles[name] = value
        else:
            # If receptacle with same name exists
            # Simply keep the value from first episode
            pass
    
    return merged_receptacles

def combine_evaluation_proposition_dependencies(ep1, ep2):
    """Combine evaluation proposition dependencies from two episodes"""
    # Number of propositions in the first episode
    n_props1 = len(ep1['evaluation_propositions'])
    
    # Initialize combined dependencies
    combined_deps = []
    has_deps = False
    
    # Check for empty lists
    both_empty = False
    
    # Process dependencies from first episode
    if 'evaluation_proposition_dependencies' in ep1 and ep1['evaluation_proposition_dependencies']:
        # Check if it's an empty list
        if isinstance(ep1['evaluation_proposition_dependencies'], list) and len(ep1['evaluation_proposition_dependencies']) == 0:
            both_empty = True
        else:
            has_deps = True
            # Copy as is
            combined_deps = copy.deepcopy(ep1['evaluation_proposition_dependencies'])
    
    # Process dependencies from second episode
    if 'evaluation_proposition_dependencies' in ep2 and ep2['evaluation_proposition_dependencies']:
        # Check if it's an empty list
        if isinstance(ep2['evaluation_proposition_dependencies'], list) and len(ep2['evaluation_proposition_dependencies']) == 0:
            # If first episode has no dependencies, keep both_empty = True
            if not has_deps:  # If first episode has no dependencies
                both_empty = True
        else:
            has_deps = True
            both_empty = False  # If second episode has dependencies, both_empty is False
            deps2 = copy.deepcopy(ep2['evaluation_proposition_dependencies'])
            
            # Process each dependency item
            for dep in deps2:
                # Adjust proposition_indices
                if 'proposition_indices' in dep:
                    dep['proposition_indices'] = [idx + n_props1 for idx in dep['proposition_indices']]
                
                # Adjust depends_on
                if 'depends_on' in dep:
                    dep['depends_on'] = [idx + n_props1 for idx in dep['depends_on']]
                
                # Add adjusted dependency
                combined_deps.append(dep)
    
    # If both episodes have empty lists
    if both_empty:
        return []
    
    # No longer adding dependencies connecting the two episodes
    
    if both_empty:
        return []
    else:
        return combined_deps if has_deps else None

def combine_episodes(ep1, ep2, combination_type, new_episode_id):
    """Combine two episodes to create a new episode"""
    combined_episode = copy.deepcopy(ep1)
    
    # Update basic information
    combined_episode['episode_id'] = str(new_episode_id)
    
    # Combine instructions
    combined_episode['instruction'] = generate_combined_instruction(ep1, ep2, combination_type)
    
    # Combine evaluation propositions
    combined_episode['evaluation_propositions'] = combine_evaluation_propositions(ep1, ep2)
    
    # Combine evaluation constraints
    combined_episode['evaluation_constraints'] = combine_evaluation_constraints(ep1, ep2)
    
    # Combine evaluation_proposition_dependencies
    dependencies = combine_evaluation_proposition_dependencies(ep1, ep2)
    
    # Always include the field regardless of dependency result (set to empty list if None)
    if dependencies is None:
        combined_episode['evaluation_proposition_dependencies'] = []
    else:
        combined_episode['evaluation_proposition_dependencies'] = dependencies
    
    # Combine metadata
    combined_episode['metadata'] = combine_metadata(ep1, ep2, combination_type)
    
    # Merge info fields
    combined_episode['info'] = merge_info_fields(ep1.get('info', {}), ep2.get('info', {}))
    
    # Merge rigid_objs fields
    combined_episode['rigid_objs'] = merge_rigid_objs(ep1.get('rigid_objs', []), ep2.get('rigid_objs', []))
    
    # Merge name_to_receptacle fields
    combined_episode['name_to_receptacle'] = merge_name_to_receptacle(
        ep1.get('name_to_receptacle', {}), 
        ep2.get('name_to_receptacle', {})
    )
    
    # Update original_data_info field as a list
    combined_episode['original_data_info'] = {
        'episode_id': [ep1['episode_id'], ep2['episode_id']],
        'instruction': [ep1.get('instruction', ''), ep2.get('instruction', '')],
        'data_path': [
            ep1.get('original_data_info', {}).get('data_path', ''),
            ep2.get('original_data_info', {}).get('data_path', '')
        ]
    }
    
    return combined_episode

def create_combined_dataset(input_file, output_file):
    """Create combined dataset with specified input and output paths"""
    # Data loading
    dataset = load_dataset(input_file)
    original_episodes = dataset['episodes']
    
    # Group episodes by scene_id and type
    scene_episodes = group_episodes_by_scene_and_type(original_episodes)
    
    # Start new episode ID
    new_episode_id = 10000  # Start from large number to avoid overlap
    
    # List for combined episodes
    combined_episodes = []
    
    # Scene combination count
    scene_combination_count = defaultdict(int)
    
    # Object overlap statistics
    object_overlap_stats = {
        'target_target': 0,
        'distractor_distractor': 0,
        'target1_distractor2': 0,
        'target2_distractor1': 0,
        'target_category_overlap': 0,
        'total_episodes_with_overlaps': 0,
        'max_overlaps': 0
    }
    
    # Receptacle overlap statistics
    receptacle_overlap_stats = {
        'total_episodes_with_overlaps': 0,
        'total_overlaps': 0,
        'max_overlaps': 0
    }
    
    # Track used episode pairs
    used_episode_pairs = set()
    
    # Track used episodes
    used_episodes = set()
    
    # Track reused episodes
    reused_episodes = set()
    
    # Scenes with insufficient episodes
    scenes_with_insufficient_episodes = {
        'OS+OS': [],
        'OS+UP': [], 
        'UP+UP': []
    }
    
    # Combination types processing order
    combination_types = [
        ("Object Semantics + Object Semantics", "OS+OS"),
        ("Object Semantics + User Pattern", "OS+UP"),
        ("User Pattern + User Pattern", "UP+UP")
    ]
    
    # Process each combination type
    for comb_type, short_type in combination_types:
        print(f"\nProcessing {comb_type} combinations...")
        
        # Process each scene
        for scene_id, type_episodes in scene_episodes.items():
            # Original episodes by type
            original_os_episodes = type_episodes.get('object_semantics', [])
            original_up_episodes = type_episodes.get('user_pattern', [])
            
            # Process based on combination type
            if comb_type == "Object Semantics + Object Semantics":
                # OS+OS combination
                if len(original_os_episodes) >= 2:
                    process_combination(
                        scene_id=scene_id,
                        ep_list1=original_os_episodes,
                        ep_list2=original_os_episodes,
                        comb_type=comb_type,
                        short_type=short_type,
                        is_same_list=True,
                        new_episode_id=new_episode_id,
                        used_episodes=used_episodes,
                        used_episode_pairs=used_episode_pairs,
                        reused_episodes=reused_episodes,
                        combined_episodes=combined_episodes,
                        scene_combination_count=scene_combination_count,
                        object_overlap_stats=object_overlap_stats,
                        receptacle_overlap_stats=receptacle_overlap_stats,
                        scenes_with_insufficient_episodes=scenes_with_insufficient_episodes
                    )
                    
                    # Update new_episode_id if combination was successful
                    if scene_combination_count[scene_id] > 0:
                        new_episode_id += 1
                else:
                    print(f"  Scene {scene_id} has fewer than 2 OS episodes.")
                    scenes_with_insufficient_episodes[short_type].append(scene_id)
            
            elif comb_type == "Object Semantics + User Pattern":
                # OS+UP combination
                if original_os_episodes and original_up_episodes:
                    process_combination(
                        scene_id=scene_id,
                        ep_list1=original_os_episodes,
                        ep_list2=original_up_episodes,
                        comb_type=comb_type,
                        short_type=short_type,
                        is_same_list=False,
                        new_episode_id=new_episode_id,
                        used_episodes=used_episodes,
                        used_episode_pairs=used_episode_pairs,
                        reused_episodes=reused_episodes,
                        combined_episodes=combined_episodes,
                        scene_combination_count=scene_combination_count,
                        object_overlap_stats=object_overlap_stats,
                        receptacle_overlap_stats=receptacle_overlap_stats,
                        scenes_with_insufficient_episodes=scenes_with_insufficient_episodes
                    )
                    
                    # Update new_episode_id if combination was successful
                    if scene_combination_count[scene_id] > 0:
                        new_episode_id += 1
                else:
                    if not original_os_episodes:
                        print(f"  Scene {scene_id} has no OS episodes.")
                    if not original_up_episodes:
                        print(f"  Scene {scene_id} has no UP episodes.")
                    scenes_with_insufficient_episodes[short_type].append(scene_id)
            
            elif comb_type == "User Pattern + User Pattern":
                # UP+UP combination
                if len(original_up_episodes) >= 2:
                    process_combination(
                        scene_id=scene_id,
                        ep_list1=original_up_episodes,
                        ep_list2=original_up_episodes,
                        comb_type=comb_type,
                        short_type=short_type,
                        is_same_list=True,
                        new_episode_id=new_episode_id,
                        used_episodes=used_episodes,
                        used_episode_pairs=used_episode_pairs,
                        reused_episodes=reused_episodes,
                        combined_episodes=combined_episodes,
                        scene_combination_count=scene_combination_count,
                        object_overlap_stats=object_overlap_stats,
                        receptacle_overlap_stats=receptacle_overlap_stats,
                        scenes_with_insufficient_episodes=scenes_with_insufficient_episodes
                    )
                    
                    # Update new_episode_id if combination was successful
                    if scene_combination_count[scene_id] > 0:
                        new_episode_id += 1
                else:
                    print(f"  Scene {scene_id} has fewer than 2 UP episodes.")
                    scenes_with_insufficient_episodes[short_type].append(scene_id)
    
    # Episode usage statistics
    episode_usage_stats = defaultdict(int)
    for pair in used_episode_pairs:
        episode_usage_stats[pair[0]] += 1
        episode_usage_stats[pair[1]] += 1
    
    # Reuse statistics
    reuse_counts = Counter(episode_usage_stats.values())
    
    # Print scene combination counts
    print("\nCombined episodes per scene:")
    for scene_id, count in sorted(scene_combination_count.items()):
        print(f"Scene {scene_id}: {count}")
    
    # Check episode counts by combination type
    type_counts = defaultdict(int)
    for ep in combined_episodes:
        if 'metadata' in ep and 'combination_type' in ep['metadata']:
            type_counts[ep['metadata']['combination_type']] += 1
    
    print("\nEpisodes by combination type:")
    for comb_type, count in type_counts.items():
        print(f"{comb_type}: {count}")
    
    # Print episode reuse statistics
    print("\nEpisode reuse statistics:")
    print(f"Total unique episodes used: {len(used_episodes)}")
    print(f"Episodes reused: {len(reused_episodes)}")
    print(f"Average uses per episode: {sum(episode_usage_stats.values()) / len(episode_usage_stats):.2f}")
    print("Usage distribution:")
    for count, num_episodes in sorted(reuse_counts.items()):
        print(f"  Episodes used {count} times: {num_episodes}")
    
    # Print object overlap statistics
    print("\nObject overlap statistics:")
    print(f"Episodes with overlaps: {object_overlap_stats['total_episodes_with_overlaps']} / {len(combined_episodes)} ({object_overlap_stats['total_episodes_with_overlaps']/len(combined_episodes)*100:.1f}%)")
    print(f"Target-target overlaps: {object_overlap_stats['target_target']}")
    print(f"Distractor-distractor overlaps: {object_overlap_stats['distractor_distractor']}")
    print(f"Episode 1 target-Episode 2 distractor overlaps: {object_overlap_stats['target1_distractor2']}")
    print(f"Episode 2 target-Episode 1 distractor overlaps: {object_overlap_stats['target2_distractor1']}")
    print(f"Target category overlaps: {object_overlap_stats['target_category_overlap']}")
    print(f"Maximum object overlaps in a single episode: {object_overlap_stats['max_overlaps']}")
    
    # Print receptacle overlap statistics
    print("\nReceptacle overlap statistics:")
    print(f"Episodes with receptacle overlaps: {receptacle_overlap_stats['total_episodes_with_overlaps']} / {len(combined_episodes)} ({receptacle_overlap_stats['total_episodes_with_overlaps']/len(combined_episodes)*100:.1f}%)")
    print(f"Total receptacle overlaps: {receptacle_overlap_stats['total_overlaps']}")
    print(f"Maximum receptacle overlaps in a single episode: {receptacle_overlap_stats['max_overlaps']}")
    
    # Report scenes with insufficient episodes
    print("\nScenes with insufficient episodes:")
    for combination_type, scenes in scenes_with_insufficient_episodes.items():
        print(f"Failed {combination_type} combinations: {len(scenes)} scenes")
        if scenes:
            print(f"  Scenes: {', '.join(str(scene) for scene in scenes)}")
    
    # Create combined dataset
    combined_dataset = copy.deepcopy(dataset)
    combined_dataset['episodes'] = combined_episodes
    
    # Save result
    save_dataset(combined_dataset, output_file)
    
    return len(combined_episodes)

def process_combination(scene_id, ep_list1, ep_list2, comb_type, short_type, is_same_list, 
                        new_episode_id, used_episodes, used_episode_pairs, reused_episodes,
                        combined_episodes, scene_combination_count, object_overlap_stats, 
                        receptacle_overlap_stats, scenes_with_insufficient_episodes):
    """Helper function for processing episode combinations"""
    print(f"  Processing Scene {scene_id} ({comb_type})...")
    
    # Step 1: Try to create combinations using only unused episodes
    combined_ep = find_valid_combination(
        scene_id, ep_list1, ep_list2, comb_type, short_type, is_same_list,
        new_episode_id, used_episodes, used_episode_pairs, 
        allow_reuse=False
    )
    
    # Step 2: If combining with unused episodes is not possible, retry including already used episodes
    if combined_ep is None:
        print(f"  Cannot create combination in Scene {scene_id} using only unused episodes. Trying with episode reuse...")
        combined_ep = find_valid_combination(
            scene_id, ep_list1, ep_list2, comb_type, short_type, is_same_list,
            new_episode_id, used_episodes, used_episode_pairs, 
            allow_reuse=True
        )
    
    # Process combination result
    if combined_ep:
        # Store used episode IDs
        ep1_id = combined_ep['original_data_info']['episode_id'][0]
        ep2_id = combined_ep['original_data_info']['episode_id'][1]
        
        # Check if episodes are being reused
        if ep1_id in used_episodes:
            reused_episodes.add(ep1_id)
        if ep2_id in used_episodes:
            reused_episodes.add(ep2_id)
        
        # Record used episodes
        used_episodes.add(ep1_id)
        used_episodes.add(ep2_id)
        
        # Add episode and update statistics
        combined_episodes.append(combined_ep)
        
        # Increment the combination count for current scene
        if scene_id in scene_combination_count:
            scene_combination_count[scene_id] += 1
        else:
            scene_combination_count[scene_id] = 1
            
        # Update object overlap statistics
        overlaps = combined_ep['metadata']['object_overlaps']['counts']
        if overlaps['total_overlap'] > 0 or overlaps['target_category_overlap'] > 0:
            object_overlap_stats['total_episodes_with_overlaps'] += 1
            object_overlap_stats['target_target'] += overlaps['target_target']
            object_overlap_stats['distractor_distractor'] += overlaps['distractor_distractor']
            object_overlap_stats['target1_distractor2'] += overlaps['target1_distractor2']
            object_overlap_stats['target2_distractor1'] += overlaps['target2_distractor1']
            object_overlap_stats['target_category_overlap'] += overlaps['target_category_overlap']
            object_overlap_stats['max_overlaps'] = max(object_overlap_stats['max_overlaps'], overlaps['total_overlap'])
        
        # Update receptacle overlap statistics
        receptacle_overlaps_info = combined_ep['metadata']['receptacle_overlaps']
        if receptacle_overlaps_info['has_overlaps']:
            receptacle_overlap_stats['total_episodes_with_overlaps'] += 1
            receptacle_overlap_stats['total_overlaps'] += receptacle_overlaps_info['overlap_count']
            receptacle_overlap_stats['max_overlaps'] = max(receptacle_overlap_stats['max_overlaps'], receptacle_overlaps_info['overlap_count'])
    else:
        print(f"  Could not create {comb_type} combination in Scene {scene_id}")
        scenes_with_insufficient_episodes[short_type].append(scene_id)

def find_valid_combination(scene_id, ep_list1, ep_list2, comb_type, short_type, is_same_list,
                          new_episode_id, used_episodes, used_episode_pairs, allow_reuse):
    """Function to find valid episode combinations"""
    
    # Find unused pairs
    valid_pairs = []
    
    # For combinations from the same list (OS+OS or UP+UP)
    if is_same_list:
        for i in range(len(ep_list1)):
            for j in range(i+1, len(ep_list2)):  # Start from i+1 to avoid duplicates
                ep1 = ep_list1[i]
                ep2 = ep_list2[j]
                pair_id = (ep1['episode_id'], ep2['episode_id'])
                rev_pair_id = (ep2['episode_id'], ep1['episode_id'])
                
                # Check if pair has already been used
                if pair_id in used_episode_pairs or rev_pair_id in used_episode_pairs:
                    continue
                
                # Check episode reuse policy
                if not allow_reuse and (ep1['episode_id'] in used_episodes or ep2['episode_id'] in used_episodes):
                    continue
                
                valid_pairs.append((ep1, ep2))
    
    # For combinations from different lists (OS+UP)
    else:
        for ep1 in ep_list1:
            for ep2 in ep_list2:
                pair_id = (ep1['episode_id'], ep2['episode_id'])
                
                # Check if pair has already been used
                if pair_id in used_episode_pairs:
                    continue
                
                # Check episode reuse policy
                if not allow_reuse and (ep1['episode_id'] in used_episodes or ep2['episode_id'] in used_episodes):
                    continue
                
                valid_pairs.append((ep1, ep2))
    
    # If there are valid pairs
    if valid_pairs:
        # Select the first pair without overlaps
        for pair in valid_pairs:
            ep1, ep2 = pair
            
            # Check for object overlaps
            object_overlaps = detect_object_overlaps(ep1, ep2)
            # Check for receptacle overlaps
            receptacle_overlaps = check_receptacle_overlaps(ep1, ep2)
            
            # Check overlap conditions
            # 1. Target category overlaps
            # 2. Target-distractor ID overlaps
            # 3. Previous overlap conditions
            max_overlap = 1 if comb_type == "Object Semantics + Object Semantics" else 2
            
            if (object_overlaps['has_target_category_overlap'] or 
                object_overlaps['has_target_distractor_overlap'] or 
                object_overlaps['counts']['total_overlap'] > max_overlap or 
                receptacle_overlaps['has_overlaps']):
                
                # Log overlap types
                overlap_types = []
                if object_overlaps['has_target_category_overlap']:
                    overlap_types.append(f"Target category overlap ({len(object_overlaps['target_category_overlap'])})")
                if object_overlaps['has_target_distractor_overlap']:
                    overlap_types.append("Target-distractor ID overlap")
                if object_overlaps['counts']['total_overlap'] > max_overlap:
                    overlap_types.append(f"Object ID overlap ({object_overlaps['counts']['total_overlap']})")
                if receptacle_overlaps['has_overlaps']:
                    overlap_types.append(f"Receptacle overlap ({receptacle_overlaps['overlap_count']})")
                
                reuse_status = "with reused episodes" if (ep1['episode_id'] in used_episodes or ep2['episode_id'] in used_episodes) else ""
                print(f"  Overlaps detected in {comb_type} combination {reuse_status}: {', '.join(overlap_types)}")
                continue
            
            # Record used pair
            used_episode_pairs.add((ep1['episode_id'], ep2['episode_id']))
            
            # Combine episodes
            combined_ep = combine_episodes(ep1, ep2, comb_type, new_episode_id)
            return combined_ep
    
    # If no valid combination found
    return None

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Combine episodes from a dataset.')
    parser.add_argument('--input', '-i', type=str, default=DEFAULT_INPUT_FILE,
                        help=f'Path to input file (default: {DEFAULT_INPUT_FILE})')
    parser.add_argument('--output', '-o', type=str, default=DEFAULT_OUTPUT_FILE,
                        help=f'Path to output file (default: {DEFAULT_OUTPUT_FILE})')
    
    args = parser.parse_args()
    
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    
    # Create combined dataset
    combined_count = create_combined_dataset(args.input, args.output)
    print(f"\nSuccessfully created {combined_count} combined episodes.") 