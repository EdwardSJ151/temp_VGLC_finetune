import torch
import gc
import traceback
import os
import time
import json
import yaml
import argparse
import wandb
from huggingface_hub import login
from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from typing import List, Dict, Any
from itertools import product
from utils.model_utils import MODEL_DATA_PREP, MODEL_CONFIG, MODEL_TRAINERS, load_model_for_family, cleanup_directories, update_run_log

parser = argparse.ArgumentParser(description="Unsloth Finetune with YAML")
parser.add_argument("--config_path", type=str, help="Path to YAML", required=True)
args = parser.parse_args()

with open(args.config_path, 'r') as f:
    config = yaml.safe_load(f)

os.environ["WANDB_WATCH"] = "all"
os.environ["WANDB_SILENT"] = "true"
mytoken = os.environ["HUGGINGFACE_TOKEN"]

wandb.login()
login(token=mytoken)

max_seq_length = 2048
dtype = None
load_in_4bit = True
seed = 3407
fp16_enabled = not is_bfloat16_supported()
bf16_enabled = is_bfloat16_supported()
dataset_num_proc = 4
packing = False
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
warmup_ratio = 0.05
learning_rate = 2e-4
logging_steps = 10
optim = "adamw_8bit"
weight_decay = 0.01
lr_scheduler_type = "linear"

wandb_settings: Dict[str, Any] = config.get('wandb_settings', {})
wandb_entity: str | None = wandb_settings.get('entity')
wandb_project: str = wandb_settings.get('project')

dataset_conf: Dict[str, Any] = config.get('dataset_config', {})
hf_path: str | None = dataset_conf.get('hf_path')
splits_to_run: List[str] = dataset_conf.get('splits_to_run', [])

print("Starting grid search...")

for current_split in splits_to_run:
    print(f"\n===== Processing Split: {current_split} =====")

    base_dataset = load_dataset(hf_path, split=current_split)

    print(f"Dataset Loaded for {current_split}")

    standardized_dataset = standardize_sharegpt(base_dataset)

    known_config_keys = {'wandb_settings', 'dataset_config'}
    for family_name, family_config in config.items():
        if family_name in known_config_keys:
            continue

        print(f"\n--- Processing Model Family: {family_name} ---")
        try:
            models_to_train: List[str] = family_config['models']
            param_grid: Dict[str, List[Any]] = family_config['params'].copy()
            family_instruction_part: str | None = family_config.get('instruction_part')
            family_response_part: str | None = family_config.get('response_part')
            lora_pairs: List[List[int]] = param_grid.pop('lora_rank_alpha_pairs')

            other_param_names: List[str] = list(param_grid.keys())
            if other_param_names:
                 other_param_value_lists: List[List[Any]] = [param_grid[name] for name in other_param_names]
                 other_param_combinations = list(product(*other_param_value_lists))
            else:
                 other_param_combinations = [{}]

        except KeyError as e:
            raise KeyError(f"Error: Missing required key {e} in config for family '{family_name}'. Skipping family.")
        except Exception as e:
            raise Exception(f"Error processing config for family '{family_name}': {e}. Skipping family.")

        for model_name in models_to_train:
            print(f"\n- Model: {model_name}")

            for r_alpha_pair in lora_pairs:
                current_r, current_lora_alpha = r_alpha_pair
                print(f"- LoRA Pair: r={current_r}, alpha={current_lora_alpha}")

                for other_param_values in other_param_combinations:
                    if other_param_names:
                        current_other_params = dict(zip(other_param_names, other_param_values))
                    else:
                        current_other_params = {}

                    run_timestamp = int(time.time())
                    print(f"- Other Params: {current_other_params}")

                    try:
                        current_num_train_epochs = current_other_params['num_train_epochs']
                        current_train_on_responses = current_other_params['train_on_responses_only']

                        resp_flag = "resp" if current_train_on_responses else "full"
                        safe_split_name = current_split.replace('-', '_').replace('/', '_')
                        run_name = f"{hf_path}_{safe_split_name}-{model_name.split('/')[-1][:15]}-{resp_flag}-r{current_r}-a{current_lora_alpha}-e{current_num_train_epochs}-{run_timestamp % 10000}"
                        output_dir = f"batch_outputs/{run_name}"
                        save_model_name = f"batch_saved_models/{run_name}"
                        os.makedirs(output_dir, exist_ok=True)
                        os.makedirs(save_model_name, exist_ok=True)

                        print(f"Run Name: {run_name}")

                        model, tokenizer = load_model_for_family(
                            family_name, model_name, max_seq_length, dtype, load_in_4bit
                        )

                        model, tokenizer = MODEL_CONFIG[family_name.lower()](
                            model, tokenizer, r=current_r, lora_alpha=current_lora_alpha,
                            random_state=seed
                        )

                        formatted_dataset = MODEL_DATA_PREP[family_name.lower()](standardized_dataset, tokenizer)

                        params_for_note = {
                            'r': current_r, 
                            'alpha': current_lora_alpha,
                            'epochs': current_num_train_epochs,
                            'train_on_responses': current_train_on_responses
                        }
                        params_str = json.dumps(params_for_note, sort_keys=True)
                        wandb_note = f"Split: {current_split}, Model: {model_name.split('/')[-1][:25]}, Params: {params_str}"
                        get_trainer = MODEL_TRAINERS[family_name.lower()]

                        max_retries = 3
                        retry_count = 0
                        success = False

                        while retry_count < max_retries and not success:
                            try:
                                wandb.init(
                                    entity=wandb_entity, 
                                    project=wandb_project, 
                                    name=run_name,
                                    notes=wandb_note
                                )

                                if family_name.lower() in ["llama3", "qwen2.5"]:
                                    trainer = get_trainer(model, tokenizer, formatted_dataset, max_seq_length)
                                else:
                                    trainer = get_trainer(model, tokenizer, formatted_dataset)

                                trainer.args.output_dir = output_dir
                                trainer.args.num_train_epochs = current_num_train_epochs
                                
                                if current_train_on_responses:
                                    trainer = train_on_responses_only(
                                        trainer, instruction_part=family_instruction_part, 
                                        response_part=family_response_part
                                    )

                                print(f"Starting training for run: {run_name}")
                                trainer.train()
                                print(f"Training finished for run: {run_name}")

                                model.save_pretrained(save_model_name)
                                tokenizer.save_pretrained(save_model_name)
                                print(f"Model saved to {save_model_name}")
                                success = True

                                run_data = {
                                    "timestamp": run_timestamp,
                                    "run_name": run_name,
                                    "model_name": model_name,
                                    "split": current_split,
                                    "params": {
                                        "r": current_r,
                                        "lora_alpha": current_lora_alpha,
                                        **current_other_params
                                    },
                                    "status": "success",
                                    "retry_count": retry_count
                                }
                                update_run_log("training_log.json", run_data)

                            except Exception as e:
                                if wandb.run:
                                    wandb.finish(exit_code=1)

                                error_msg = str(e)
                                traceback_msg = traceback.format_exc()
                                retry_count += 1
                                
                                run_data = {
                                    "timestamp": run_timestamp,
                                    "run_name": run_name,
                                    "model_name": model_name,
                                    "split": current_split,
                                    "params": {
                                        "r": current_r,
                                        "lora_alpha": current_lora_alpha,
                                        **current_other_params
                                    },
                                    "status": "failed",
                                    "error": error_msg,
                                    "traceback": traceback_msg,
                                    "retry_count": retry_count
                                }
                                update_run_log("training_log.json", run_data)

                                print(f"Error during training (attempt {retry_count}/{max_retries}): {e}")
                                print(f"Traceback: {traceback_msg}")
                                
                                if 'trainer' in locals(): 
                                    del trainer

                                if retry_count < max_retries:
                                    print(f"Retrying training")
                                    time.sleep(1)
                                else:
                                    print(f"Max retries ({max_retries}) reached.")
                                    cleanup_directories(output_dir, save_model_name)

                    except Exception as e:
                        print(f"Error during run: {e}")
                        print(f"Traceback: {traceback.format_exc()}")
                        if wandb.run: wandb.finish(exit_code=1)
                    finally:
                        if wandb.run: wandb.finish()
                        
                        if 'tokenizer' in locals(): del tokenizer
                        if 'trainer' in locals(): 
                            if family_name.lower() == "gemma":
                                if hasattr(trainer, 'model'):
                                    if hasattr(trainer.model, 'language_model'):
                                        trainer.model.language_model = None
                                    if hasattr(trainer.model, 'lm_head'):
                                        trainer.model.lm_head = None
                            del trainer
                        if 'formatted_dataset' in locals(): del formatted_dataset
                        if 'model' in locals():
                            if family_name.lower() == "gemma":
                                if hasattr(model, 'language_model'):
                                    model.language_model = None
                                if hasattr(model, 'lm_head'):
                                    model.lm_head = None
                            model.cpu()
                            del model
                            gc.collect()
                        
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                        print("Run Complete\n")

print("\nFinish")