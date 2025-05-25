import sys
import os
import gc
from IPython.display import display
from utils.inference_utils import extract_level_representation, fix_level_format, fix_level_format_extra
from utils.create_img import convert_kidicarus_to_png, convert_loderunner_to_png, convert_mario_to_png, convert_rainbowisland_to_png
from unsloth import FastLanguageModel
import torch
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import io
from PIL import Image
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
import json
import yaml
import argparse
from utils.metrics import SampledLevelEvaluator
from utils.inference_utils import load_model_by_type, generate_with_model, game_settings

parser = argparse.ArgumentParser(description="Run batch inference with configuration from a YAML file.")
parser.add_argument("--config_file", type=str, required=True)
args = parser.parse_args()

with open(args.config_file, 'r') as f:
    config = yaml.safe_load(f)

models = config.get("models", {})
temperatures = config.get("temperatures", [0.7, 1.0, 1.2, 1.5, 2.0])
num_of_samples = config.get("num_of_samples", 5)
game_type = config.get("game") # options: "mario", "loderunner", "kidicarus", "rainbowisland"
orientation = config.get("orientation",) #options: "horizontal", "vertical"

prompt = "Create a level"
output_pdf = f"level_generation_results_{game_type}_{orientation}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
max_seq_length = 2048
dtype = None
load_in_4bit = True

json_output = f"level_generation_results_{game_type}_{orientation}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
results_data = []
evaluator = SampledLevelEvaluator()

with PdfPages(output_pdf) as pdf:
    for model_type, model_paths in models.items():
        print(f"Processing model type: {model_type}")
        
        for model_path in model_paths:
            print(f"Processing model: {model_path}")
            
            try:
                model, tokenizer = load_model_by_type(
                    model_path=model_path,
                    model_type=model_type,
                    max_seq_length=max_seq_length,
                    dtype=dtype,
                    load_in_4bit=load_in_4bit
                )
                
                for temp in temperatures:
                    print(f"Running with temperature: {temp}")
                    
                    for sample_idx in range(num_of_samples):
                        print(f"Generating sample {sample_idx+1}/{num_of_samples}")
                        
                        response = generate_with_model(
                            model=model,
                            tokenizer=tokenizer,
                            prompt=prompt,
                            model_type=model_type,
                            temperature=temp
                        )

                        level = extract_level_representation(
                            response[0],
                            model_type=model_type,
                            orientation=orientation,
                        )

                        fixed_level = fix_level_format_extra(
                            level,
                            empty_space=game_settings[game_type]["empty_space"],
                            line_quantity=game_settings[game_type]["line_quantity"],
                            column_quantity=game_settings[game_type]["column_quantity"],
                            enforce_shape="both",
                            orientation="horizontal",
                            add_ground=game_settings[game_type]["add_ground"]
                        )

                        fixed_level = fixed_level.replace("|", "\n")

                        expected_output_size = game_settings[game_type]["expected_output_size"]
                        level_without_separators = level.replace("\n", "").replace("|", "")
                        diff_percentage = SampledLevelEvaluator.calculate_generation_diff(
                            expected_output_size,
                            level_without_separators
                        )

                        result_data = {
                            "model_type": model_type,
                            "model_path": os.path.basename(model_path),
                            "temperature": temp,
                            "sample_index": sample_idx + 1,
                            "level": fixed_level,
                            "metrics": {
                                "expected_size": expected_output_size,
                                "actual_size": len(level_without_separators),
                                "size_diff_percentage": diff_percentage
                            }
                        }
                        results_data.append(result_data)
                        
                        convert_function = game_settings[game_type]["convert_function"]
                        tiles_dir = game_settings[game_type]["tiles_dir"]
                        if tiles_dir:
                            img, _, _ = convert_function(fixed_level, tiles_dir=tiles_dir)
                        else:
                            img, _, _ = convert_function(fixed_level)
                        
                        plt.figure(figsize=(12, 10))
                        
                        metadata = (
                            f"Model Type: {model_type}\n"
                            f"Model: {os.path.basename(model_path)}\n"
                            f"Temperature: {temp}\n"
                            f"Sample: {sample_idx+1}/{num_of_samples}\n"
                            f"Size Diff: {diff_percentage:.2f}%\n"
                            f"Level:\n{fixed_level}"
                        )
                        
                        plt.subplot(2, 1, 1)
                        plt.text(0.05, 0.95, metadata, fontsize=8, va='top', 
                                 family='monospace', transform=plt.gca().transAxes)
                        plt.axis('off')
                        plt.subplot(2, 1, 2)
                        plt.imshow(img)
                        plt.axis('off')
                        plt.title(f"Generated Level")
                        plt.tight_layout()

                        pdf.savefig()
                        plt.close()
                        
                        print(f"Sample {sample_idx+1} completed")
                    
                    print(f"Completed temperature {temp}")
                
                print(f"Completed model {model_path}")
                
                model.cpu()
                del model
                gc.collect()
                del tokenizer
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                
            except Exception as e:
                if model:
                    model.cpu()
                    del model
                    gc.collect()
                if tokenizer:
                    del tokenizer
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

                plt.figure(figsize=(8.5, 11))
                error_info = (
                    f"Error processing model: {model_path}\n"
                    f"Model type: {model_type}\n\n"
                    f"Error: {str(e)}"
                )
                plt.text(0.5, 0.5, error_info, fontsize=12, ha='center', va='center', color='red')
                plt.axis('off')
                pdf.savefig()
                plt.close()
                print(f"Error with model {model_path}: {str(e)}")
    
    plt.figure(figsize=(8.5, 11))
    
    total_models = sum(len(paths) for paths in models.values())
    
    info = (
        f"Level Generation Results\n\n"
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Model Types: {', '.join(models.keys())}\n"
        f"Total Models: {total_models}\n"
        f"Temperatures: {temperatures}\n"
        f"Samples per combination: {num_of_samples}\n"
        f"Game type: {game_type}\n"
        f"Total samples: {total_models * len(temperatures) * num_of_samples}"
    )
    plt.text(0.5, 0.5, info, fontsize=12, ha='center', va='center')
    plt.axis('off')
    pdf.savefig()
    plt.close()

with open(json_output, 'w') as f:
    json.dump(results_data, f, indent=2)

print(f"PDF saved to {output_pdf}")
print(f"JSON saved to {json_output}")