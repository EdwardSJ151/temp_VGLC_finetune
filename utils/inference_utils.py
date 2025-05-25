from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
from transformers import TextStreamer
from utils.create_img import convert_kidicarus_to_png, convert_loderunner_to_png, convert_mario_to_png, convert_rainbowisland_to_png

game_settings = {
    "mario": {
        "empty_space": "-",
        "line_quantity": 14,
        "column_quantity": 50,
        "convert_function": convert_mario_to_png,
        "tiles_dir": './assets/mario',
        "add_ground": "X",
        "expected_output_size": 700
    },
    "loderunner": {
        "empty_space": ".",
        "line_quantity": 22,
        "column_quantity": 32,
        "convert_function": convert_loderunner_to_png,
        "tiles_dir": './assets/lode_runner',
        "add_ground": None,
        "expected_output_size": 704
    },
    "kidicarus": {
        "empty_space": "-",
        "line_quantity": 20,
        "column_quantity": 16,
        "convert_function": convert_kidicarus_to_png,
        "tiles_dir": './assets/kid_icarus',
        "add_ground": None,
        "expected_output_size": 320
    },
    "rainbowisland": {
        "empty_space": ".",
        "line_quantity": 35,
        "column_quantity": 32,
        "convert_function": convert_rainbowisland_to_png,
        "tiles_dir": './assets/rainbow_island',
        "add_ground": None,
        "expected_output_size": 1120
    }
}

def load_model_by_type(model_path, model_type, max_seq_length=2048, dtype=None, load_in_4bit=True):
    """Load model based on model type"""
    if model_type in ["llama-3", "qwen-2.5", "llama3", "qwen2.5"]:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
    elif model_type in ["qwen-3", "qwen3"]:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
    elif model_type in ["gemma-3", "gemma3"]:
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            load_in_8bit=False,
            full_finetuning=False,
        )
        FastModel.for_inference(model)
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="gemma-3",
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model, tokenizer

def generate_with_model(model, tokenizer, prompt, model_type, temperature=0.7):

    if model_type in ["gemma-3", "gemma3"]:
        tokenizer = get_chat_template(
            tokenizer,
            chat_template = "gemma-3",
        )
        messages = [{
            "role": "user",
            "content": [{
                "type" : "text",
                "text" : prompt,
            }]
        }]
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True, # Must add for generation
        )
        outputs = model.generate(
            **tokenizer([text], return_tensors = "pt").to("cuda"),
            max_new_tokens = 4096,
            temperature = temperature, top_p = 0.95, top_k = 64,
        )


    elif model_type in ["qwen-3", "qwen3"]:
        messages = [
            {"role" : "user", "content" : prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = True,
            enable_thinking = False,
        )

        outputs = model.generate(
            **tokenizer(text, return_tensors = "pt").to("cuda"),
            max_new_tokens = 4096,
            temperature = temperature, top_p = 0.8, top_k = 20,
            streamer = TextStreamer(tokenizer, skip_prompt = True),
        )
    
    elif model_type in ["llama-3", "qwen-2.5", "llama3", "qwen2.5"]:
        messages = [
            {"role": "user", "content": prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True, # Must add for generation
            return_tensors = "pt",
        ).to("cuda")


        outputs = model.generate(input_ids = inputs, max_new_tokens = 4096, use_cache = True,
                                 temperature = temperature, min_p = 0.1)


    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return tokenizer.batch_decode(outputs)

###################################################################

def extract_level_representation(llm_output, model_type="llama-3", orientation="horizontal", separator="\n", empty_space='-'):
    if isinstance(llm_output, list):
        llm_output = llm_output[0]

    level_content = ""
    if model_type in ["llama-3", "llama3"]:
        parts = llm_output.split("<|start_header_id|>assistant<|end_header_id|>")
        if len(parts) > 1:
            assistant_section = parts[-1]

            if "<|eot_id|>" in assistant_section:
                level_content = assistant_section.split("<|eot_id|>")[0].strip()
            else:
                level_content = assistant_section.strip()
        else:
            level_content = llm_output.strip()

    elif model_type in ["gemma-3", "gemma3"]:
        parts = llm_output.split("<start_of_turn>model")

        if len(parts) > 1:
            model_section = parts[-1]

            if "<end_of_turn>" in model_section:
                level_content = model_section.split("<end_of_turn>")[0].strip()
            else:
                level_content = model_section.strip()
        else:
            level_content = llm_output.strip()

    elif model_type in ["qwen-2.5", "qwen2.5"]:
        parts = llm_output.split("<|im_start|>assistant")

        if len(parts) > 1:
            assistant_block = parts[-1]

            if "<|im_end|>" in assistant_block:
                level_content = assistant_block.split("<|im_end|>")[0].strip()
            else:
                level_content = assistant_block.strip()
        else:
            level_content = llm_output.strip()

    elif model_type in ["qwen-3", "qwen3"]:
        parts = llm_output.split("<|im_start|>assistant")

        if len(parts) > 1:
            assistant_block = parts[-1]

            if "<|im_end|>" in assistant_block:
                content = assistant_block.split("<|im_end|>")[0].strip()

                if "<think>" in content and "</think>" in content:
                    think_parts = content.split("</think>")
                    if len(think_parts) > 1:
                        level_content = think_parts[-1].strip()
                    else:
                        level_content = content
                else:
                    level_content = content
            else:
                level_content = assistant_block.strip()
        else:
            level_content = llm_output.strip()

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # print("New level content:")
    # print(level_content)

    if "|" in level_content and "\n" not in level_content:
        separator = "|"
    elif "\n" in level_content and "|" not in level_content:
        separator = "\n"

    if orientation == "vertical":
        level_content = VerticalLevel.reconstruct_level_from_vertical_bar(level_content, separator)
        
    return level_content


def fix_level_format(level_str, orientation="horizontal", separator="\n", empty_space='-'):
    if orientation == "vertical":
        level_str = VerticalLevel.reconstruct_level_from_vertical_bar(level_str, separator)
    
    lines = level_str.split('\n')

    line_lengths = [len(line) for line in lines]

    changed = True
    while changed:
        changed = False
        max_length = max(line_lengths)
        longest_lines_indices = [i for i, length in enumerate(line_lengths) if length == max_length]

        lines_trimmed = False
        for idx in longest_lines_indices:
            line = lines[idx]
            if line and line[-1] == empty_space:
                lines[idx] = line[:-1]
                line_lengths[idx] -= 1
                lines_trimmed = True
                changed = True

        if not lines_trimmed:
            break

    max_length = max(line_lengths)

    for i in range(len(lines)):
        if line_lengths[i] < max_length:
            padding_char = empty_space

            lines[i] = lines[i] + (padding_char * (max_length - line_lengths[i]))

    return '\n'.join(lines)

##############################################################################################

def fix_level_format_extra(level_str, orientation="horizontal", separator="\n", empty_space='-', 
                     line_quantity=None, column_quantity=None, enforce_shape=None, add_ground=None,
                     use_original_logic_on_column=False):

    if "|" in level_str and "\n" not in level_str:
        separator = "|"
    elif "\n" in level_str and "|" not in level_str:
        separator = "\n"

    if orientation == "vertical":
        level_str = VerticalLevel.reconstruct_level_from_vertical_bar(level_str, separator)
    
    lines = level_str.split(separator)
    
    # Handle line quantity enforcement
    if enforce_shape in ["line", "both"] and line_quantity is not None:
        if len(lines) > line_quantity:
            # Remove from top (beginning of list)
            lines = lines[-line_quantity:]
        elif len(lines) < line_quantity:
            # Add empty lines at top
            empty_line = empty_space * (max(len(line) for line in lines) if lines else column_quantity or 0)
            lines = [empty_line] * (line_quantity - len(lines)) + lines
    
    # Handle column quantity enforcement
    column_adjusted = False
    if enforce_shape in ["column", "both"] and column_quantity is not None:
        column_adjusted = True
        for i in range(len(lines)):
            if len(lines[i]) > column_quantity:
                lines[i] = lines[i][:column_quantity]
            elif len(lines[i]) < column_quantity:
                if i == len(lines) - 1 and add_ground is not None:
                    lines[i] = lines[i] + (add_ground * (column_quantity - len(lines[i])))
                else:
                    lines[i] = lines[i] + (empty_space * (column_quantity - len(lines[i])))
    
    if (enforce_shape is None or use_original_logic_on_column) and not column_adjusted:
        line_lengths = [len(line) for line in lines]
        
        changed = True
        while changed:
            changed = False
            max_length = max(line_lengths)
            longest_lines_indices = [i for i, length in enumerate(line_lengths) if length == max_length]
            
            lines_trimmed = False
            for idx in longest_lines_indices:
                line = lines[idx]
                if line and line[-1] == empty_space:
                    lines[idx] = line[:-1]
                    line_lengths[idx] -= 1
                    lines_trimmed = True
                    changed = True
            
            if not lines_trimmed:
                break
        
        max_length = max(line_lengths)
        for i in range(len(lines)):
            if line_lengths[i] < max_length:
                padding_char = add_ground if i == len(lines) - 1 and add_ground is not None else empty_space
                lines[i] = lines[i] + (padding_char * (max_length - line_lengths[i]))
    
    return separator.join(lines)

##############################################################################################

class VerticalLevel:
    @staticmethod
    def reconstruct_level_from_vertical_bar(vertical_bar_str, separator="\n"):
        if not vertical_bar_str:
            return None

        vertical_columns = vertical_bar_str.split(separator)
        num_cols = len(vertical_columns)

        if num_cols == 0 or not vertical_columns[0]:
            return None

        num_rows = len(vertical_columns[0])
        if num_rows == 0:
            return None 

        reconstructed_rows = []
        for i in range(num_rows): 
            current_row_chars = []
            for j in range(num_cols): 
                vertical_char_index = num_rows - 1 - i
                if j < len(vertical_columns) and vertical_char_index < len(vertical_columns[j]):
                    char = vertical_columns[j][vertical_char_index]
                    current_row_chars.append(char)

            reconstructed_rows.append("".join(current_row_chars))

        return "\n".join(reconstructed_rows)
    

def determine_game_and_orientation(path: str) -> tuple[str, str]:
    path = path.lower()
    
    game_type = None
    if "mario" in path:
        game_type = "mario"
    elif "icarus" in path or "kid" in path:
        game_type = "kidicarus"
    elif "runner" in path or "lode" in path:
        game_type = "loderunner"
    elif "rainbow" in path or "island" in path:
        game_type = "rainbowisland"
    
    orientation = "horizontal"
    if "vertical" in path:
        orientation = "vertical"
    
    return game_type, orientation