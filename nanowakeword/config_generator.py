# Copyright 2025 Arcosoph. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import yaml
import logging
import json


logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')

def clamp(value, min_val, max_val):
    """Helper function to keep a value within a specified range."""
    return max(min_val, min(value, max_val))

class ConfigGenerator:
    def __init__(self, stats):
        self.stats = stats
        self.config = {}
        
        self.C = {
            'recommended_minimum_h_pos': 0.5,
            'target_neg_pos_ratio': 3.0,
            'target_effective_data_hours': 20.0,
            'min_augmentation_rounds': 10,
            'max_augmentation_rounds': 15,
            'model_complexity_scaler': 2.0,
            'base_lr': 5e-5,
            'lr_size_sensitivity': 0.1,
            'dropout_risk_scaler': 0.5,
            'steps_per_effective_hour': 1000,
            'min_steps': 10000,
            'max_steps': 40000
        }



    def generate(self, data_generation_is_planned=False):
        print("Applying intelligent formulas for harmonization and configuration...")

       # --- Step 1: Initial Data Analysis ---
        H_pos_initial = self.stats.get('H_pos', 0.0)
        H_neg_initial = self.stats.get('H_neg', 0.0)
        H_noise = self.stats.get('H_noise', 0.0)
        A_noise = self.stats.get('A_noise', 0.0)
        N_rir = self.stats.get('N_rir', 0)
        
        # --- Step 2: Create a data harmonization plan (this is always an output) ---
        self.config['data_generation_plan'] = {
            'generate_positive_hours': 0.0,
            'generate_negative_hours': 0.0
        }
        H_pos_after_synth = H_pos_initial
        if H_pos_initial < self.C['recommended_minimum_h_pos']:
            hours_to_gen = self.C['recommended_minimum_h_pos'] - H_pos_initial
            self.config['data_generation_plan']['generate_positive_hours'] = hours_to_gen
            H_pos_after_synth = self.C['recommended_minimum_h_pos']
            logging.info(f"Planning to generate {hours_to_gen:.2f}h of positive data.")

        H_neg_after_synth = H_neg_initial
        if (H_neg_initial / (H_pos_after_synth + 1e-9)) < self.C['target_neg_pos_ratio']:
            target_neg_hours = H_pos_after_synth * self.C['target_neg_pos_ratio']
            hours_to_gen = target_neg_hours - H_neg_initial
            if hours_to_gen > 0:
                self.config['data_generation_plan']['generate_negative_hours'] = hours_to_gen
                H_neg_after_synth = target_neg_hours
                logging.info(f"Planning to generate {hours_to_gen:.2f}h of negative data.")

       # --- Step 3: Determining the basis of calculation ---
        if data_generation_is_planned:
            base_hours_for_calculation = H_pos_after_synth + H_neg_after_synth
            logging.info("Calculating config for FUTURE dataset (including planned synthetic data).")
        else:
            base_hours_for_calculation = H_pos_initial + H_neg_initial
            logging.info("Calculating config for CURRENT dataset (real data only).")
      
            
        # --- Step 4: Calculate data volume and augmentation round ---
        dynamic_target_hours = clamp(5 + 10 * np.log1p(base_hours_for_calculation), 5.0, 50.0)
        
        if base_hours_for_calculation > 0.01:
            required_multiplier = dynamic_target_hours / base_hours_for_calculation
        else:
            required_multiplier = self.C['max_augmentation_rounds'] * 2
            
        calculated_rounds = int(round(clamp(
            required_multiplier, self.C['min_augmentation_rounds'], self.C['max_augmentation_rounds']
        )))
        self.config['augmentation_rounds'] = calculated_rounds
        logging.info(f"Setting 'augmentation_rounds' to {calculated_rounds}.")
        
        effective_data_volume = base_hours_for_calculation * calculated_rounds
        
        

       # step 5
        quality_score = (1 - clamp(A_noise, 0, 1)) + clamp(N_rir / 500, 0, 1)
        normalized_quality = quality_score / 2

        # 
        base_steps = int(effective_data_volume * self.C['steps_per_effective_hour'])
        adjustment_factor = 1.1 - (0.2 * normalized_quality)
        calculated_steps = int(base_steps * adjustment_factor)
        self.config['steps'] = int(clamp(calculated_steps, self.C['min_steps'], self.C['max_steps']))

        # 
        model_complexity = clamp(np.log10(effective_data_volume + 1) * self.C['model_complexity_scaler'], 1.0, 4.0)
        self.config['model_complexity_score'] = model_complexity
        self.config['n_blocks'] = int(round(model_complexity))
        layer_size = 64 * (2 ** (self.config['n_blocks'] - 1))
        self.config['layer_size'] = int(clamp(layer_size, 64, 512))

        # 
        base_lr = self.C['base_lr']
        size_factor = (effective_data_volume / 20)**self.C['lr_size_sensitivity']
        noise_factor = (1 - clamp(A_noise, 0, 1))**2
        max_lr = base_lr * clamp(size_factor, 0.8, 2.0) * clamp(noise_factor, 0.5, 1.0)
        self.config['learning_rate_max'] = max_lr
        self.config['learning_rate_base'] = max_lr / 10

        # 
        model_capacity = self.config['n_blocks'] * (self.config['layer_size'] ** 2)
        dataset_size_proxy = effective_data_volume * 3600
        overfitting_risk = model_capacity / (dataset_size_proxy * 1000 + 1e-6)
        dropout_prob = clamp(0.5 + (overfitting_risk * self.C['dropout_risk_scaler']), 0.2, 0.7)
        self.config['dropout_prob'] = dropout_prob

        # RIR and Noise Probability
        self.config['rir_probability'] = clamp(1 - (1 / (1 + N_rir * 0.1)), 0.0, 0.8)
        self.config['background_noise_probability'] = clamp(1 - (1 / (1 + H_noise * 2)), 0.2, 0.9)

        # 
        self.config['max_negative_weight'] = clamp(5.0 / model_complexity, 1.5, 10.0)

        # 
        num_cycles = clamp(effective_data_volume / 25, 2, 4)
        total_cycle_steps = self.config['steps'] / num_cycles
        self.config['clr_step_size_up'] = int(total_cycle_steps * 0.4)
        self.config['clr_step_size_down'] = int(total_cycle_steps * 0.6)

        # 
        noise_need = 1 - clamp(H_noise / 0.5, 0, 1)
        dist_noise = int(10 + 15 * noise_need)
        neg_speech_need = 1 - clamp(H_neg_after_synth / (H_pos_after_synth * self.C['target_neg_pos_ratio']), 0, 1)
        remaining_percentage = 100 - dist_noise
        dist_neg_speech = int((remaining_percentage * 0.75) * (0.8 + 0.2 * neg_speech_need))
        dist_pos = 100 - dist_noise - dist_neg_speech
        self.config['source_distribution'] = {
            'positive': dist_pos,
            'negative_speech': dist_neg_speech,
            'pure_noise': dist_noise
        }

        logging.info("Intelligent configuration complete.")
        return self.config


    def save_config(self, path, base_config_path):
        """
        Saves the generated configuration by updating a base config file.

        Args:
            path (str): The output path for the new, complete config file.
            base_config_path (str): Path to the base user config file (contains paths etc.).
        """
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        
        base_config.update(self.config)
        
        with open(path, 'w') as f:
            yaml.dump(base_config, f, default_flow_style=False, sort_keys=False)
        
        logging.info(f"Complete configuration saved to {path}")



if __name__ == '__main__':
   
    print("Running standalone test for ConfigGenerator...")

    test_stats = {
        'H_pos': 0.49626836805556174,
        'H_neg': 1.4672359722222177,
        'H_noise': 0.23194444444444445,
        'A_noise': 0.1166294522654251,
        'N_rir': 417
    }


    # test_stats = {
    #     'H_pos': 0.12,
    #     'H_neg': 0.34,
    #     'H_noise': 0.05,
    #     'A_noise': 0.02,
    #     'N_rir': 15
    # }

    # test_stats = {
    # 'H_pos': 0.67,
    # 'H_neg': 1.15,
    # 'H_noise': 0.42,
    # 'A_noise': 0.29,
    # 'N_rir': 95
    # }

# generate(self, data_generation_is_planned=False)
    print("\n--- Input Stats ---")
    print(test_stats)
    
    generator = ConfigGenerator(test_stats)
    generated_config = generator.generate()

    print("\n--- Generated Config & Data Plan ---")
   
    print(json.dumps(generated_config, indent=2))
    print("------------------------------------")