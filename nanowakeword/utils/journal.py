# ==============================================================================
#  NanoWakeWord: Lightweight, Intelligent Wake Word Detection
#  Copyright 2025 Arcosoph. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Project: https://github.com/arcosoph/nanowakeword
# ==============================================================================

import os
import json
from .logger import print_info

def format_change_value(value):
    """Formats values for display, keeping it concise."""
    if isinstance(value, list):
        # For lists (like blueprints), just show that it has changed without printing the whole list
        return f"[list len={len(value)}]"
    if isinstance(value, dict):
        return "[dict]"
    return value

def update_training_journal(base_output_dir, model_name, metrics, current_config):
    """
    The master journal with smart grouping for nested parameters.
    - Long parameters like 'audio_processing.autotune...' are grouped under 'audio_processing'.
    - The cell for a group only shows the sub-parameters that have changed.
    """
    try:
        journal_path = os.path.join(base_output_dir, "training_journal.md")
        cache_dir = os.path.join(base_output_dir,".cache", "journal_cache")
        history_db_path = os.path.join(cache_dir, "training_history.json")
        
        os.makedirs(cache_dir, exist_ok=True)

        PERMANENT_COLUMNS = ["Name", "Stb Loss", "APC", "ANC", "Time"]

        HEADER_MAP = {
            "Model Name": "Name", 
            "Stable Loss": "Stb Loss",
            "Avg. Pos Conf": "APC",  
            "Avg. Neg Conf": "ANC",  
            "Train Time": "Time"
        }


        EXCLUDED_PARAMS = {
            'output_dir', 'generate_clips', 'transform_clips', 'train_model', 
            'overwrite', 'resume', 'onnx_opset_version', 'force_verify', 
            'show_training_summary', 'model_name', 'enable_journaling',
            'feature_manifest', 'custom_negative_per_phrase', 'custom_negative_phrases',
            'rir_paths', 'background_paths', 'negative_data_path', 'positive_data_path', 
        }

        # 1. Process and group the current run's configuration
        top_level_config = {}
        grouped_config = {}
        for key, value in current_config.items():
            if key in EXCLUDED_PARAMS:
                continue
            if '.' in key:
                group_name, sub_key = key.split('.', 1)
                if group_name not in grouped_config:
                    grouped_config[group_name] = {}
                grouped_config[group_name][sub_key] = value
            else:
                top_level_config[key] = value

        # 2. Prepare the display row for the current run
        current_display_row = {HEADER_MAP[k]: v for k, v in metrics.items()}
        current_display_row[HEADER_MAP["Model Name"]] = f"**{model_name}**"

        # 3. Load history and apply "Show on Change" logic
        all_runs_history = []
        if os.path.exists(history_db_path):
            with open(history_db_path, 'r', encoding='utf-8') as f:
                try: all_runs_history = json.load(f)
                except json.JSONDecodeError: all_runs_history = []

        if not all_runs_history:
            # First run, nothing to compare, so display row is just the metrics
            pass
        else:
            last_run = all_runs_history[-1]
            last_top_level = last_run.get('top_level_config', {})
            last_grouped = last_run.get('grouped_config', {})

            # Compare top-level parameters
            for param, value in top_level_config.items():
                if param not in last_top_level or last_top_level[param] != value:
                    current_display_row[param] = value
            
            # Compare grouped parameters
            for group, sub_params in grouped_config.items():
                changes_in_group = []
                last_group_params = last_grouped.get(group, {})
                for sub_key, sub_value in sub_params.items():
                    if sub_key not in last_group_params or last_group_params[sub_key] != sub_value:
                        changes_in_group.append(f"`{sub_key}`: `{format_change_value(sub_value)}`")
                
                if changes_in_group:
                    current_display_row[group] = "; ".join(changes_in_group)

        # 4. Update and save the history database
        new_history_entry = {
            "display_row": current_display_row,
            "top_level_config": top_level_config,
            "grouped_config": grouped_config
        }
        all_runs_history.append(new_history_entry)
        
        with open(history_db_path, 'w', encoding='utf-8') as f:
            json.dump(all_runs_history, f, indent=4)

        # 5. Generate and write the Markdown table
        all_columns = PERMANENT_COLUMNS[:]
        for run in all_runs_history:
            for key in run['display_row'].keys():
                if key not in all_columns:
                    all_columns.append(key)
        
        header = "| " + " | ".join(all_columns) + " |"
        separator = "| " + " | ".join([":---"] * len(all_columns)) + " |"
        
        rows = []
        for run in all_runs_history:
            row_str = "| " + " | ".join(str(run['display_row'].get(col, "")) for col in all_columns) + " |"
            rows.append(row_str)

        with open(journal_path, 'w', encoding='utf-8') as f:
            f.write("# NanoWakeWord Training Journal\n\n")
            f.write(header + "\n")
            f.write(separator + "\n")
            f.write("\n".join(rows) + "\n")
        
        print_info(f"Master training journal updated successfully at '{journal_path}'")

    except Exception as e:
        import traceback
        print(f"[Journal Error] Failed to update journal. Details: {e}")
        traceback.print_exc()