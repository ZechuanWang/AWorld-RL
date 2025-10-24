#!/usr/bin/env python3
"""
BFCL_v3 Data Preprocessing Script

This script converts BFCL_v3 multi-turn API interaction data to verl format for RL training.
Supports all four BFCL_v3 dataset types: base, long_context, miss_func, miss_param.

Key Features:
- Loads tool schemas from multi_turn_func_doc directory
- Handles missing function scenarios with dynamic function availability
- Processes multi-turn conversation structures
- Converts to verl-compatible format with proper ground truth handling
- Supports data splitting with reproducible random seeding
"""

import argparse
import json
import os
import random
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from string import Template
from bfcl_eval.constants.default_prompts import DEFAULT_SYSTEM_PROMPT_FOR_CHAT_MODEL,DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING
from bfcl_eval.utils import load_file, dump_json, load_json
from bfcl_eval.constants.category_mapping import MULTI_TURN_FUNC_DOC_FILE_MAPPING,VERSION_PREFIX,TEST_FILE_MAPPING,TEST_COLLECTION_MAPPING
from bfcl_eval.constants.eval_config import PROMPT_PATH,MULTI_TURN_FUNC_DOC_PATH,POSSIBLE_ANSWER_PATH
# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
import pdb
import traceback

def convert_bfclv3_to_verl_format(
    entry: Dict[str, Any],
    ground_truth_entry: Dict[str, Any],
    dataset_type : str
) -> Dict[str, Any]:
    """Convert a BFCL_v3 sample to verl format."""
    # Extract basic information
    id = entry.get("id", "")
    involved_classes = entry.get("involved_classes", [])
    assert involved_classes is not None, "Involved classes must be provided"

    # Get functions for this entry
    entry["function"] = []
    for func_collection in involved_classes:
        # func_doc is a list of dict
        func_doc = load_file(
            MULTI_TURN_FUNC_DOC_PATH / MULTI_TURN_FUNC_DOC_FILE_MAPPING[func_collection]
        )
        entry["function"].extend(func_doc)

    # Handle Miss Func category; we need to remove the holdout function doc
    if "missed_function" in entry:
        for turn_index, missed_func_names in entry["missed_function"].items():
            entry["missed_function"][turn_index] = []
            for missed_func_name in missed_func_names:
                for i, func_doc in enumerate(entry["function"]):
                    if func_doc["name"] == missed_func_name:
                        # Add the missed function doc to the missed_function list
                        entry["missed_function"][turn_index].append(func_doc)
                        # Remove it from the function list
                        entry["function"].pop(i)
                        break

    # Generate system prompt
    functions = json.dumps(entry["function"],ensure_ascii=False)
    system_prompt = DEFAULT_SYSTEM_PROMPT_FOR_CHAT_MODEL.substitute(functions=functions)
    
    # Process question turns and handle missed functions
    question_turns = entry.get("question")
    assert question_turns is not None, "Questions should not be None! "
    processed_questions = []
    
    # Extract and store the first question for initial prompt
    first_question = None
    if "missed_function" in entry:
        miss_func_turn = list(entry["missed_function"].keys())[0]
        assert isinstance(miss_func_turn,str), "Missed function turn should be string"
        miss_func_turn = int(miss_func_turn)
    for turn_idx, turn in enumerate(question_turns):
        content: Optional[str] = None
        if "missed_function" in entry:
            if turn_idx != miss_func_turn:
                content = turn[0]["content"]
            else:
                # Empty turn - check if we need to insert missed functions
                turn_str = str(turn_idx)
                missed_funcs = entry["missed_function"].get(turn_str)
                assert isinstance(missed_funcs,list), "missed functions should be list"
        
                # Create prompt with additional functions
                additional_functions_json = json.dumps(missed_funcs, ensure_ascii=False)
                content = DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING.format(
                    functions=additional_functions_json
                )
        else:
            content = turn[0]["content"]
        assert content is not None, "Content should not be None after miss function handle!"
        # Store first valid question as initial prompt, subsequent ones go to processed_questions
        if turn_idx == 0:
            first_question = content
        else:
            processed_questions.append(content)

    assert first_question is not None, "First question should not be None!"
    
    # Get ground truth data
    assert ground_truth_entry.get("id") == id, "Gound Truth id should be same!"
    ground_truth = ground_truth_entry.get("ground_truth")
    assert ground_truth is not None, "Ground Truth should not be None!"
    # Build the verl format with JSON serialization for complex objects
    verl_entry = {
        "data_source": dataset_type,  # Mark the specific dataset type
        "prompt": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user", 
                "content": first_question
            }
        ],
        "ability": "multi_turn_function_calling",
        "reward_model": {
            "style": "interaction", 
            # "interaction_type": "multi_turn_vm",
            "ground_truth": ground_truth  # Ground truth function sequence
        },
        "extra_info": {
            "split": "train",  # Will be overridden based on the split
            "index": entry.get("id", "unknown"),
            "original_id": entry.get("id", "unknown"),
            "dataset_type": dataset_type,  # Additional dataset type info
            # Serialize complex objects as JSON strings to avoid parquet issues
            "interaction_kwargs": {
                "name": "multi_turn_tool_call",
                "id":id,
                "initial_config": json.dumps(entry.get("initial_config", {}),ensure_ascii=False),
                "involved_classes": involved_classes,
                "ground_truth": ground_truth,
                "processed_question": processed_questions,  # Processed question turns with missed function prompts
                "question":entry.get("question")
            }
        }
    }
    return verl_entry


def split_data_by_ratio(samples: List[Dict[str, Any]], train_ratio: float = 0.5, val_ratio: float = 0.125, test_ratio: float = 0.375) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split data into train/val/test sets with specified ratios."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Shuffle samples
    shuffled_samples = samples.copy()
    random.shuffle(shuffled_samples)
    
    total_samples = len(shuffled_samples)
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)
    
    train_samples = shuffled_samples[:train_end]
    val_samples = shuffled_samples[train_end:val_end]
    test_samples = shuffled_samples[val_end:]
    
    return train_samples, val_samples, test_samples


def process_all_bfclv3_files(
    output_dir: str,
    split_data: bool = True
):
    """Process all BFCL_v3 files in the directory and optionally split into train/val/test."""
    categories = TEST_COLLECTION_MAPPING.get("multi_turn", [])
    assert categories is not None, "Categories cannot be None!"
    
    
    # Collect all samples from all files
    all_entries = []
    processed_files = []
    train_entries = []
    val_entries = []
    test_entries = []
    for category in categories:
        category_file = TEST_FILE_MAPPING.get(category, "")
        assert category is not None, "File path cannot be None!"
        category_file_path = os.path.join(PROMPT_PATH, category_file)
        category_entries = []
        if os.path.exists(category_file_path):
            print(f"=== Processing {category_file} ===")

            # Load ground truth for the current file
            answer_dir = os.path.join(POSSIBLE_ANSWER_PATH, category_file)
            ground_truth = load_file(answer_dir)
            print(f"Loaded ground truth for {len(ground_truth)} samples from {category}")
            print(f"Processing {category} dataset: {category_file_path}")
            
            # Read BFCLV3 data - these files are actually JSONL format (one JSON per line)
            category_data = load_file(category_file_path)
            # Convert each sample to verl format
            for i, entry in enumerate(category_data):
                try:
                    ground_truth_entry = None
                    for g in ground_truth:
                        if g.get("id") == entry.get("id"):
                            ground_truth_entry = g
                            break
                    assert ground_truth_entry is not None, "Ground truth for this entry should not be None!"
                    verl_entry = convert_bfclv3_to_verl_format(entry, ground_truth_entry, category)
                    verl_entry["extra_info"]["index"] = f"{category}_{i}"
                    verl_entry["extra_info"]["original_file"] = category_file
                    category_entries.append(verl_entry)
                    all_entries.append(verl_entry)
                except Exception as e:
                    traceback.print_exc()    
                    pdb.set_trace()
                    print(f"Error processing sample {i} from {category_file}: {e}")
                    continue
            
            processed_files.append(category_file)
            print(f"Converted {len(category_data)} samples from {category_data}")
            if split_data and len(category_entries) > 0:
                train, val, test = split_data_by_ratio(category_entries)
                splits = [
                    (f"{VERSION_PREFIX}_train", train),
                    (f"{VERSION_PREFIX}_val", val), 
                    (f"{VERSION_PREFIX}_test", test)
                ]
                for split_name, entries in splits:
                    if len(entries) > 0:
                        # Update split information in each sample
                        for entry in entries:
                            entry["extra_info"]["split"] = split_name.replace(f"{VERSION_PREFIX}_", "")
                train_entries.extend(train)
                val_entries.extend(val)
                test_entries.extend(test)
        else:
            print(f"Warning: File not found: {category_file}")
    
    print(f"\n=== Total samples collected: {len(all_entries)} ===")

    if len(train_entries) > 0:
            output_file_parquet = os.path.join(output_dir, "bfcl_train.parquet")
            df = pd.DataFrame(train_entries)
            df.to_parquet(output_file_parquet, index=False)
            print(f"Saved {len(train_entries)} samples to {output_file_parquet}")
            output_file_json = os.path.join(output_dir, "bfcl_train.json")
            dump_json(train_entries, output_file_json)
        
    if len(val_entries) > 0:
            output_file_parquet = os.path.join(output_dir, "bfcl_val.parquet")
            df = pd.DataFrame(val_entries)
            df.to_parquet(output_file_parquet, index=False)
            print(f"Saved {len(val_entries)} samples to {output_file_parquet}")
            output_file_json = os.path.join(output_dir, "bfcl_val.json")
            dump_json(val_entries, output_file_json)
    if len(test_entries) > 0:
            output_file_parquet = os.path.join(output_dir, "bfcl_test.parquet")
            df = pd.DataFrame(test_entries)
            df.to_parquet(output_file_parquet, index=False)
            print(f"Saved {len(test_entries)} samples to {output_file_parquet}")
            output_file_json = os.path.join(output_dir, "bfcl_test.json")
            dump_json(test_entries, output_file_json)

    if len(all_entries) > 0:
        output_file_parquet = os.path.join(output_dir, "bfcl_all.parquet")
        df = pd.DataFrame(all_entries)
        df.to_parquet(output_file_parquet, index=False)
        print(f"Saved {len(all_entries)} samples to {output_file_parquet}")
        output_file_json = os.path.join(output_dir, "bfcl_all.json")
        dump_json(all_entries, output_file_json)

    
    return all_entries

def main():
    parser = argparse.ArgumentParser(description="Convert BFCLV3 data to verl format")
    parser.add_argument("--output_dir", default="./processed_data/bfcl_v3/", help="Output directory for processed data")
    parser.add_argument("--no_split", action="store_true", help="Don't split data into train/val/test, save as single file")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
        
    total_samples = process_all_bfclv3_files(
                args.output_dir,
                split_data=not args.no_split
            )
    print(f"\nSuccessfully processed all files with {len(total_samples)} total samples")

if __name__ == "__main__":
    main()
   