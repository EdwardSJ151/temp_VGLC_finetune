import os
import shutil
import json
from unsloth import FastLanguageModel, FastModel
from typing import Dict, Any, Tuple, List, Callable
from datasets import Dataset
from unsloth.chat_templates import standardize_data_formats, get_chat_template
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported


def cleanup_directories(output_dir: str, save_model_dir: str):
    """Remove directories created for a failed run"""
    for dir_path in [output_dir, save_model_dir]:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"Cleaned up directory: {dir_path}")
            except Exception as e:
                print(f"Error cleaning up directory {dir_path}: {e}")

def update_run_log(log_file: str, run_data: dict):
    """Update the JSON log file with new run data"""
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log = json.load(f)
        else:
            log = {"runs": []}
        
        log["runs"].append(run_data)
        
        with open(log_file, 'w') as f:
            json.dump(log, f, indent=2)
    except Exception as e:
        print(f"Error updating log file: {e}")

###############################################################################3

def load_model_for_family(family_name: str, model_name: str, max_seq_length: int, dtype=None, load_in_4bit=True):
    """Load the appropriate model based on model family"""
    from unsloth import FastLanguageModel, FastModel
    
    if family_name.lower() == "gemma3":
        # Gemma needs FastModel, not FastLanguageModel
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
    elif family_name.lower() in ["llama3", "qwen2.5", "qwen3"]:
        # These models use FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
    else:
        raise ValueError(f"Unsupported model family: {family_name}. Must be one of: gemma3, llama3, qwen2.5, qwen3")
        
    return model, tokenizer

# -------------------
# Gemma3 Functions
# -------------------

def gemma3_data_prep(dataset: Dataset, tokenizer: Any) -> Dataset:

    # Standardize data formats to ensure consistency
    standardized_dataset = standardize_data_formats(dataset)

    print(f"GEMMA3 DEBUG - Standardized dataset first item: {standardized_dataset[0]}")

    # Apply chat template
    def apply_chat_template(examples):
        texts = tokenizer.apply_chat_template(examples["conversations"], tokenize=False)
        return {"text": texts}
    
    formatted_dataset = standardized_dataset.map(apply_chat_template, batched=True)
    return formatted_dataset


def gemma3_model_config(
    model: Any, 
    tokenizer: Any, 
    r: int = 64, 
    lora_alpha: int = 64,
    random_state: int = 3407
) -> Tuple[Any, Any]:
    """
    Configure a Gemma3 model with appropriate parameters.
    """
    # Use FastModel for Gemma models
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers = False,  # Turn off for just text
        finetune_language_layers = True,  # Should leave on
        finetune_attention_modules = True,  # Attention good for training
        finetune_mlp_modules = True,  # Should leave on always
        
        r = r,                # LoRA rank
        lora_alpha = lora_alpha,  # Recommended alpha == r at least
        lora_dropout = 0,     # Optimized setting
        bias = "none",        # Optimized setting
        random_state = random_state,
    )
    
    # Set the appropriate chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "gemma-3",
    )
    
    return model, tokenizer


def get_gemma3_trainer(model, tokenizer, dataset):
    """Get SFT trainer configured for Gemma3 models"""
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_ratio=0.05,
            num_train_epochs=1,
            # max_steps = 300,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="wandb",
        ),
    )

# -------------------
# Llama3 Functions
# -------------------

def llama3_data_prep(dataset: Dataset, tokenizer: Any) -> Dataset:
    """
    Prepare data for Llama3 models.
    
    Llama3 format uses header-based conversation style.
    """
    # Standardize for ShareGPT format
    standardized_dataset = standardize_data_formats(dataset)
    
    # Apply formatting function
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) 
                for convo in convos]
        return {"text": texts}
    
    formatted_dataset = standardized_dataset.map(formatting_prompts_func, batched=True)
    return formatted_dataset


def llama3_model_config(
    model: Any, 
    tokenizer: Any, 
    r: int = 64, 
    lora_alpha: int = 64,
    random_state: int = 3407
) -> Tuple[Any, Any]:
    """
    Configure a Llama3 model with appropriate parameters.
    """
    model = FastLanguageModel.get_peft_model(
        model,
        r = r,  # Choose any number > 0! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = lora_alpha,
        lora_dropout = 0,  # Supports any, but = 0 is optimized
        bias = "none",     # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth",  # Uses 30% less VRAM
        random_state = random_state,
        use_rslora = False,   # We support rank stabilized LoRA
        loftq_config = None,  # And LoftQ
    )
    
    # Set the appropriate chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3.1",
    )
    
    return model, tokenizer


def get_llama3_trainer(model, tokenizer, dataset, max_seq_length=2048):
    """Get SFT trainer configured for Llama3 models"""
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_ratio=0.05,
            num_train_epochs=1,
            # max_steps = 300,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="wandb",
        ),
    )

# -------------------
# Qwen2.5 Functions
# -------------------

def qwen2_5_data_prep(dataset: Dataset, tokenizer: Any) -> Dataset:
    """
    Prepare data for Qwen2.5 models.
    
    Qwen2.5 uses im_start/im_end markers for conversation formatting.
    """
    # Standardize for ShareGPT format
    standardized_dataset = standardize_data_formats(dataset)
    
    # Apply formatting function
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) 
                for convo in convos]
        return {"text": texts}
    
    formatted_dataset = standardized_dataset.map(formatting_prompts_func, batched=True)
    return formatted_dataset


def qwen2_5_model_config(
    model: Any, 
    tokenizer: Any, 
    r: int = 64, 
    lora_alpha: int = 64,
    random_state: int = 3407
) -> Tuple[Any, Any]:
    """
    Configure a Qwen2.5 model with appropriate parameters.
    """
    model = FastLanguageModel.get_peft_model(
        model,
        r = r,  # Choose any number > 0! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = lora_alpha,
        lora_dropout = 0,  # Supports any, but = 0 is optimized
        bias = "none",     # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth",  # Uses 30% less VRAM
        random_state = random_state,
        use_rslora = False,   # We support rank stabilized LoRA
        loftq_config = None,  # And LoftQ
    )
    
    # Set the appropriate chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "qwen-2.5",
    )
    
    return model, tokenizer


def get_qwen2_5_trainer(model, tokenizer, dataset, max_seq_length=2048):
    """Get SFT trainer configured for Qwen2.5 models"""
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_ratio=0.05,
            num_train_epochs=1,
            # max_steps = 300,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="wandb",
        ),
    )

# -------------------
# Qwen3 Functions
# -------------------

def qwen3_data_prep(dataset: Dataset, tokenizer: Any) -> Dataset:
    """
    Prepare data for Qwen3 models.
    
    Qwen3 uses im_start/im_end markers with potential 'think' sections in assistant responses.
    """
    # Standardize for ShareGPT format
    standardized_dataset = standardize_data_formats(dataset)
    
    # Get the conversations with chat template applied
    conversations = tokenizer.apply_chat_template(
        standardized_dataset["conversations"],
        tokenize=False,
    )
    
    # Convert to dataset format
    from pandas import Series
    from datasets import Dataset
    dataset = Dataset.from_pandas(Series(conversations, name="text").to_frame())
    
    return dataset


def qwen3_model_config(
    model: Any, 
    tokenizer: Any, 
    r: int = 64, 
    lora_alpha: int = 64,
    random_state: int = 3407
) -> Tuple[Any, Any]:
    """
    Configure a Qwen3 model with appropriate parameters.
    """
    model = FastLanguageModel.get_peft_model(
        model,
        r = r,  # Choose any number > 0! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = lora_alpha,
        lora_dropout = 0,  # Supports any, but = 0 is optimized
        bias = "none",     # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth",  # Uses 30% less VRAM
        random_state = random_state,
        use_rslora = False,   # We support rank stabilized LoRA
        loftq_config = None,  # And LoftQ
    )
    
    return model, tokenizer


def get_qwen3_trainer(model, tokenizer, dataset):
    """Get SFT trainer configured for Qwen3 models"""
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_ratio=0.05,
            num_train_epochs=1,
            # max_steps = 300,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="wandb",
        ),
    )


# Function mapping for easy lookup
MODEL_DATA_PREP = {
    "gemma3": gemma3_data_prep,
    "llama3": llama3_data_prep,
    "qwen2.5": qwen2_5_data_prep,
    "qwen3": qwen3_data_prep
}

MODEL_CONFIG = {
    "gemma3": gemma3_model_config,
    "llama3": llama3_model_config,
    "qwen2.5": qwen2_5_model_config,
    "qwen3": qwen3_model_config
}

MODEL_TRAINERS = {
    "gemma3": get_gemma3_trainer,
    "llama3": get_llama3_trainer,
    "qwen2.5": get_qwen2_5_trainer,
    "qwen3": get_qwen3_trainer
}