#!/usr/bin/env python3
"""
Inference script for Qwen2.5-VL, Gemma-3-27B-IT, InternVL3, and LLaVA models using the recommended pipeline.
Supports two modes:
1. Amazon product title generation from images (all models)
2. Image prompt generation from product descriptions (all models)

Requirements:
  pip install git+https://github.com/huggingface/transformers accelerate
  pip install qwen-vl-utils[decord]
  pip install torch pillow
  pip install lmdeploy  # For InternVL3 models
  pip install peft  # For PEFT/LoRA adapter models
"""
import json
import logging
from pathlib import Path
import argparse
import torch
import os
import re
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration, Gemma3ForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

# Import langchain for JSON parsing
try:
    from langchain_core.output_parsers import JsonOutputParser
    from langchain.schema import OutputParserException
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("langchain not available. Will fall back to regex parsing only.")

# Import PEFT for LoRA adapter models
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("peft not available. LoRA adapter models will not work.")

# Import lmdeploy for InternVL3 models only if needed
try:
    from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig, ChatTemplateConfig
    from lmdeploy.vl import load_image
    LMDEPLOY_AVAILABLE = True
except ImportError:
    LMDEPLOY_AVAILABLE = False
    logging.warning("lmdeploy not available. InternVL3 models will not work.")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def is_peft_model(model_path: str) -> bool:
    """Check if the model path contains a PEFT adapter"""
    adapter_config_path = Path(model_path) / "adapter_config.json"
    return adapter_config_path.exists()

def get_available_categories():
    """Get list of available Amazon categories"""
    categories = [
        "All_Beauty", "Toys_and_Games", "Video_Games", "Home_and_Kitchen",
        "Electronics", "Industrial_and_Scientific", "Office_Products", 
        "Musical_Instruments", "Arts_Crafts_and_Sewing"
    ]
    return categories

def get_model_type(model_name: str) -> str:
    """Determine if the model is Qwen2.5-VL (vision), Gemma-3 (vision), InternVL3 (vision), or LLaVA (vision)"""
    if "qwen" in model_name.lower():
        return "qwen_vl"
    elif "gemma" in model_name.lower():
        return "gemma_vl"  # Gemma-3-27B-IT supports vision
    elif "internvl" in model_name.lower():
        return "internvl3_vl"  # InternVL3 supports vision
    elif "llava" in model_name.lower():
        return "llava_vl"  # LLaVA supports vision
    else:
        # Default to qwen_vl for backward compatibility
        return "qwen_vl"

def get_model_size(model_name: str) -> str:
    """Extract model size from model name"""
    model_name_lower = model_name.lower()
    if "27b" in model_name_lower or "27b-instruct" in model_name_lower:
        return "27b"
    elif "12b" in model_name_lower or "12b-instruct" in model_name_lower:
        return "12b"
    elif "32b" in model_name_lower or "32b-instruct" in model_name_lower:
        return "32b"
    elif "72b" in model_name_lower or "72b-instruct" in model_name_lower:
        return "72b"
    elif "78b" in model_name_lower or "78b-instruct" in model_name_lower:
        return "78b"
    elif "34b" in model_name_lower or "34b-it" in model_name_lower:
        return "34b"
    elif "7b" in model_name_lower or "7b-it" in model_name_lower:
        return "7b"
    elif "8b" in model_name_lower or "8b-instruct" in model_name_lower:
        return "8b"
    elif "4b" in model_name_lower:
        return "4b"
    else:
        # Default size if not specified
        return "unknown"

def get_output_dir(model_name: str, mode: str) -> Path:
    """Get output directory based on model type, size and mode"""
    model_type = get_model_type(model_name)
    model_size = get_model_size(model_name)
    
    if mode == "title_bench":
        if model_type == "qwen_vl":
            return Path(f'./outputs_bench_qwen_vl_{model_size}_titles')
        elif model_type == "gemma_vl":
            return Path(f'./outputs_bench_gemma_vl_{model_size}_titles')
        elif model_type == "internvl3_vl":
            return Path(f'./outputs_bench_internvl3_vl_{model_size}_titles')
        elif model_type == "llava_vl":
            return Path(f'./outputs_bench_llava_vl_{model_size}_titles')
        else:
            return Path(f'./outputs_bench_unknown_model_{model_size}_titles')
    elif mode == "title_rec":
        if model_type == "qwen_vl":
            return Path(f'./outputs_qwen_rec_vl_{model_size}_titles')
        elif model_type == "gemma_vl":
            return Path(f'./outputs_gemma_rec_vl_{model_size}_titles')
        elif model_type == "internvl3_vl":
            return Path(f'./outputs_internvl3_rec_vl_{model_size}_titles')
        elif model_type == "llava_vl":
            return Path(f'./outputs_llava_rec_vl_{model_size}_titles')
        else:
            return Path(f'./outputs_unknown_rec_model_{model_size}_titles')
    elif mode == "title":
        if model_type == "qwen_vl":
            return Path(f'./outputs_qwen_vl_{model_size}_titles')
        elif model_type == "gemma_vl":
            return Path(f'./outputs_gemma_vl_{model_size}_titles')
        elif model_type == "internvl3_vl":
            return Path(f'./outputs_internvl3_vl_{model_size}_titles')
        elif model_type == "llava_vl":
            return Path(f'./outputs_llava_vl_{model_size}_titles')
        else:
            return Path(f'./outputs_unknown_model_{model_size}_titles')
    elif mode == "prompt_rec":
        if model_type == "qwen_vl":
            return Path(f'./outputs_rec_qwen_vl_{model_size}_prompts')
        elif model_type == "gemma_vl":
            return Path(f'./outputs_rec_gemma_vl_{model_size}_prompts')
        elif model_type == "internvl3_vl":
            return Path(f'./outputs_rec_internvl3_vl_{model_size}_prompts')
        elif model_type == "llava_vl":
            return Path(f'./outputs_rec_llava_vl_{model_size}_prompts')
        else:
            return Path(f'./outputs_rec_unknown_model_{model_size}_prompts')
    elif mode == "prompt_bench":
        if model_type == "qwen_vl":
            return Path(f'./outputs_bench_qwen_vl_{model_size}_prompts')
        elif model_type == "gemma_vl":
            return Path(f'./outputs_bench_gemma_vl_{model_size}_prompts')
        elif model_type == "internvl3_vl":
            return Path(f'./outputs_bench_internvl3_vl_{model_size}_prompts')
        elif model_type == "llava_vl":
            return Path(f'./outputs_bench_llava_vl_{model_size}_prompts')
        else:
            return Path(f'./outputs_bench_unknown_model_{model_size}_prompts')
    elif mode == "prompt":
        if model_type == "qwen_vl":
            return Path(f'./outputs_qwen_vl_{model_size}_prompts')
        elif model_type == "gemma_vl":
            return Path(f'./outputs_gemma_vl_{model_size}_prompts')
        elif model_type == "internvl3_vl":
            return Path(f'./outputs_internvl3_vl_{model_size}_prompts')
        elif model_type == "llava_vl":
            return Path(f'./outputs_llava_vl_{model_size}_prompts')
        else:
            return Path(f'./outputs_unknown_model_{model_size}_prompts')
    else:
        return Path(f'./outputs_unknown_{model_size}')

def parse_llm_json(raw_text: str) -> dict:
    """
    Parse JSON from LLM output. First try langchain JsonOutputParser,
    then fall back to regex extraction if that fails.
    """
    # First try: Use langchain JsonOutputParser if available
    if LANGCHAIN_AVAILABLE:
        json_parser = JsonOutputParser()
        try:
            parsed_result = json_parser.parse(raw_text)
            print(f"[DEBUG] LangChain parsing successful")
            return parsed_result
        except OutputParserException as e:
            print(f"[DEBUG] LangChain parsing failed: {e}, falling back to regex")
        except Exception as e:
            print(f"[DEBUG] LangChain parsing error: {e}, falling back to regex")
    
    # Second try: Regex-based extraction (original method)
    print(f"[DEBUG] Using regex parsing as fallback")
    # 1. Remove all ```json or ``` markers
    text = re.sub(r'```(?:json)?', '', raw_text)
    # 2. Remove leading/trailing whitespace
    text = text.strip()
    # 3. Extract content between first { ... }
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        raise ValueError("No JSON object found")
    json_str = m.group(0)
    # 4. Final load
    return json.loads(json_str)

def load_model(model_name: str, gpu_id: int):
    """Load model for inference. Supports PEFT adapters and regular models."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    model_type = get_model_type(model_name)
    
    # Check if this is a PEFT adapter model
    if is_peft_model(model_name):
        if not PEFT_AVAILABLE:
            raise ImportError("peft is required for LoRA adapter models.")
        
        logging.info(f"Loading PEFT adapter model from {model_name}")
        
        # Load the base model first
        if model_type == "qwen_vl":
            # Use official Qwen model instead of problematic unsloth version
            logging.info("Loading official Qwen2.5-VL-7B-Instruct as base model...")
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                device_map="auto",
                torch_dtype=torch.float16,
                quantization_config={
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": torch.float16,
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_quant_type": "nf4"
                }
            )
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        else:
            raise ValueError(f"PEFT adapter not supported for model type: {model_type}")
        
        # Load the PEFT adapter
        model = PeftModel.from_pretrained(base_model, model_name)
        return processor, model
    
    # Regular model loading (original logic)
    if model_type == "internvl3_vl":
        if not LMDEPLOY_AVAILABLE:
            raise ImportError("lmdeploy is required for InternVL3 models.")
        pipe = pipeline(
            model_name,
            backend_config=TurbomindEngineConfig(session_len=16384, tp=1),
            chat_template_config=ChatTemplateConfig(model_name='internvl2_5')
        )
        return pipe, None
    elif model_type == "qwen_vl":
        processor = AutoProcessor.from_pretrained(model_name)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config={
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4"
            }
        )
        return processor, model
    elif model_type == "gemma_vl":
        processor = AutoProcessor.from_pretrained(model_name)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config={
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.bfloat16,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4"
            }
        ).eval()
        return processor, model
    elif model_type == "llava_vl":
        processor = LlavaNextProcessor.from_pretrained(model_name)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config={
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4"
            }
        )
        return processor, model
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def infer_product_title(processor, model, image_path: Path, max_new_tokens: int = 128, model_name: str = "", session=None):
    """Run inference on an image to get product title. Returns (parsed_dict, raw_output)"""
    item_id = image_path.stem.split('.')[0]
    model_type = get_model_type(model_name)
    
    if model_type == "internvl3_vl":
        prompt = (
            "You are an Amazon product title generation assistant. "
            "Given only the product image, generate a suitable Amazon product title. "
            "Generate a JSON object with only the key: 'title'. "
            "Make sure the output is a valid JSON object that can be parsed. "
            "IMPORTANT: Please output in English only. "
            "The title should be descriptive, professional, and suitable for an Amazon listing. "
            "For example, if the image shows a stainless steel water bottle, you might output:\n"
            "{\n"
            "  \"title\": \"HydroFlow 24oz Stainless Steel Insulated Water Bottle with Leak-Proof Design\"\n"
            "}"
        )
        
        # Load image using lmdeploy
        image = load_image(str(image_path.resolve()))
        gen_config = GenerationConfig(top_k=40, top_p=0.8, temperature=0.8, max_new_tokens=max_new_tokens)
        
        try:
            sess = processor.chat((prompt, image), gen_config=gen_config, session=session)
            result = sess.response.text.strip()
            print(f"[DEBUG] InternVL3 raw output: {result}")
            try:
                parsed = parse_llm_json(result)
                parsed['image'] = item_id
                # Print the generated title
                if 'title' in parsed:
                    print(f"[TITLE OUTPUT] Item {item_id}: {parsed['title']}")
                return parsed, result, sess
            except Exception as e:
                print(f"[ERROR] Failed to parse JSON for item {item_id}: {e}")
                return {
                    'image': item_id,
                    'title': 'error'
                }, f"Error: {str(e)} | Raw: {result}", sess
        except Exception as e:
            print(f"[ERROR] Fatal error processing item {item_id}: {e}")
            return {
                'image': item_id,
                'title': 'error'
            }, f"Fatal error: {str(e)}", session
    
    if model_type == "qwen_vl":
        # Qwen2.5-VL vision processing
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str(image_path.resolve())
                    },
                    {
                    "type": "text",
                    "text": (
                        "You are an Amazon product title generation assistant. "
                        "Given only the product image, generate a suitable Amazon product title. "
                        "Generate a JSON object with only the key: 'title'. "
                        "Make sure the output is a valid JSON object that can be parsed. "
                        "IMPORTANT: Please output in English only. The output must not be empty, blank, or newline — directly give the answer."
                        "The title should be descriptive, professional, and suitable for an Amazon listing. "
                        "For example, if the image shows a stainless steel water bottle, you might output:\n"
                        "{\n"
                        "  \"title\": \"HydroFlow 24oz Stainless Steel Insulated Water Bottle with Leak-Proof Design\"\n"
                        "}\n"
                        "Output:"
                    )
                    }
                ]
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)
    elif model_type == "gemma_vl":
        # Gemma-3-27B-IT vision processing
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str(image_path.resolve())
                    },
                    {
                        "type": "text",
                        "text": (
                            "You are an Amazon product title generation assistant. "
                            "Given only the product image, generate a suitable Amazon product title. "
                            "Generate a JSON object with only the key: 'title'. "
                            "Make sure the output is a valid JSON object that can be parsed. "
                            "IMPORTANT: Please output in English only. "
                            "The title should be descriptive, professional, and suitable for an Amazon listing. "
                            "For example, if the image shows a stainless steel water bottle, you might output:\n"
                            "{\n"
                            "  \"title\": \"HydroFlow 24oz Stainless Steel Insulated Water Bottle with Leak-Proof Design\"\n"
                            "}"
                        )
                    }
                ]
            }
        ]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
    elif model_type == "llava_vl":
        # LLaVA vision processing
        from PIL import Image
        image = Image.open(image_path.resolve())
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are an Amazon product title generation assistant. "
                            "Given only the product image, generate a suitable Amazon product title. "
                            "Generate a JSON object with only the key: 'title'. "
                            "Make sure the output is a valid JSON object that can be parsed. "
                            "IMPORTANT: Please output in English only. "
                            "The title should be descriptive, professional, and suitable for an Amazon listing. "
                            "For example, if the image shows a stainless steel water bottle, you might output:\n"
                            "{\n"
                            "  \"title\": \"HydroFlow 24oz Stainless Steel Insulated Water Bottle with Leak-Proof Design\"\n"
                            "}"
                        )
                    },
                    {
                        "type": "image"
                    }
                ]
            }
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    with torch.no_grad():
        if model_type == "gemma_vl":
            input_len = inputs["input_ids"].shape[-1]
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            generated_ids_trimmed = [generated_ids[0][input_len:]]
        else:
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
    result = output_text[0].strip()
    print(f"[DEBUG] Model output: {result}")
    try:
        parsed = parse_llm_json(result)
        parsed['image'] = item_id
        # Print the generated title
        if 'title' in parsed:
            print(f"[TITLE OUTPUT] Item {item_id}: {parsed['title']}")
        return parsed, result, None
    except Exception as e:
        print(f"[ERROR] Failed to parse JSON for item {item_id}: {e}")
        # On any error, return error fields and the raw output or error message
        return {
            'image': item_id,
            'title': 'error'
        }, f"Error: {str(e)} | Raw: {result}", None

def infer_image_prompt(processor, model, product_description: str, max_new_tokens: int = 256, model_name: str = "", session=None):
    """Generate an image prompt from product description. Returns (parsed_dict, raw_output)"""
    model_type = get_model_type(model_name)
    
    if model_type == "internvl3_vl":
        prompt = (
            "You are a professional product photographer and prompt engineer. "
            "Given a product description, generate a detailed prompt for creating a product cover image. "
            "The prompt should be suitable for a diffusion model to generate a high-quality product photo. "
            "Generate a JSON object with only the key: 'prompt'. "
            "Make sure the output is a valid JSON object that can be parsed. "
            "IMPORTANT: Please output in English only. "
            "The prompt should be descriptive and professional for commercial product photography. "
            "CRITICAL: The prompt must be within 77 tokens maximum. "
            "Keep it concise but descriptive.\n\n"
            f"Product description: {product_description}\n\n"
            "Example output format:\n"
            "{\n"
            "  \"prompt\": \"professional product photography of a luxury watch on marble surface, soft lighting, high-end commercial style\"\n"
            "}"
        )
        gen_config = GenerationConfig(top_k=40, top_p=0.8, temperature=0.8, max_new_tokens=max_new_tokens)
        
        try:
            sess = processor.chat(prompt, gen_config=gen_config, session=session)
            result = sess.response.text.strip()
            print(f"[DEBUG] InternVL3 raw output: {result}")
            try:
                parsed = parse_llm_json(result)
                parsed['description'] = product_description
                # Print the generated prompt
                if 'prompt' in parsed:
                    print(f"[PROMPT OUTPUT] Description '{product_description[:50]}...': {parsed['prompt']}")
                return parsed, result, sess
            except Exception as e:
                print(f"[ERROR] Failed to parse JSON for prompt generation: {e}")
                return {
                    'description': product_description,
                    'prompt': 'error'
                }, f"Error: {str(e)} | Raw: {result}", sess
        except Exception as e:
            print(f"[ERROR] Fatal error generating prompt: {e}")
            return {
                'description': product_description,
                'prompt': 'error'
            }, f"Fatal error: {str(e)}", session
    
    if model_type == "qwen_vl":
        # Qwen2.5-VL processing
        messages = [
            {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": (
                        "You are a professional product photographer and prompt engineer. "
                        "Given a product description, generate a detailed prompt for creating a product cover image. "
                        "The prompt should be suitable for a diffusion model to generate a high-quality product photo. "
                        "Generate a JSON object with only the key: 'prompt'. "
                        "Make sure the output is a valid JSON object that can be parsed. "
                        "IMPORTANT: Please output in English only. "
                        "The prompt should be descriptive and professional for commercial product photography. "
                        "CRITICAL: The prompt must be within 77 tokens maximum. "
                        "Keep it concise but descriptive.\n\n"
                        f"Product description: {product_description}\n\n"
                        "Example output format:\n"
                        "{\n"
                        "  \"prompt\": \"professional product photography of a luxury watch on marble surface, soft lighting, high-end commercial style\"\n"
                        "}"
                    )
                    }
                ]
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            padding=True,
            return_tensors="pt"
        ).to(model.device)
    elif model_type == "gemma_vl":
        # Gemma-3-27B-IT processing
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a professional product photographer and prompt engineer. "
                            "Given a product description, generate a detailed prompt for creating a product cover image. "
                            "The prompt should be suitable for a diffusion model to generate a high-quality product photo. "
                            "Generate a JSON object with only the key: 'prompt'. "
                            "Make sure the output is a valid JSON object that can be parsed. "
                            "IMPORTANT: Please output in English only. "
                            "The prompt should be descriptive and professional for commercial product photography. "
                            "CRITICAL: The prompt must be within 77 tokens maximum. "
                            "Keep it concise but descriptive.\n\n"
                            f"Product description: {product_description}\n\n"
                            "Example output format:\n"
                            "{\n"
                            "  \"prompt\": \"professional product photography of a luxury watch on marble surface, soft lighting, high-end commercial style\"\n"
                            "}"
                        )
                    }
                ]
            }
        ]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
    elif model_type == "llava_vl":
        # LLaVA processing
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a professional product photographer and prompt engineer. "
                            "Given a product description, generate a detailed prompt for creating a product cover image. "
                            "The prompt should be suitable for a diffusion model to generate a high-quality product photo. "
                            "Generate a JSON object with only the key: 'prompt'. "
                            "Make sure the output is a valid JSON object that can be parsed. "
                            "IMPORTANT: Please output in English only. "
                            "The prompt should be descriptive and professional for commercial product photography. "
                            "CRITICAL: The prompt must be within 77 tokens maximum. "
                            "Keep it concise but descriptive.\n\n"
                            f"Product description: {product_description}\n\n"
                            "Example output format:\n"
                            "{\n"
                            "  \"prompt\": \"professional product photography of a luxury watch on marble surface, soft lighting, high-end commercial style\"\n"
                            "}"
                        )
                    }
                ]
            }
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(text=prompt, return_tensors="pt").to(model.device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    with torch.no_grad():
        if model_type == "gemma_vl":
            input_len = inputs["input_ids"].shape[-1]
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            generated_ids_trimmed = [generated_ids[0][input_len:]]
        else:
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
    result = output_text[0].strip()
    print(f"[DEBUG] Model output: {result}")
    try:
        parsed = parse_llm_json(result)
        parsed['description'] = product_description
        # Print the generated prompt
        if 'prompt' in parsed:
            print(f"[PROMPT OUTPUT] Description '{product_description[:50]}...': {parsed['prompt']}")
        return parsed, result, None
    except Exception as e:
        print(f"[ERROR] Failed to parse JSON for prompt generation: {e}")
        return {
            'description': product_description,
            'prompt': 'error'
        }, f"Error: {str(e)} | Raw: {result}", None

def process_category_images(category_name: str, model_name: str, gpu_id: int, max_items: int = 1000, mode: str = "title"):
    """Process images for a specific category to generate titles"""
    processor, model = load_model(model_name, gpu_id)
    OUTPUT_DIR = get_output_dir(model_name, mode)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Category-specific output file
    out_file = OUTPUT_DIR / f'{category_name}_titles_gpu{gpu_id}.jsonl'
    
    # Check existing processed items
    processed = set()
    if out_file.exists():
        for line in out_file.read_text(encoding='utf-8').splitlines():
            try:
                rec = json.loads(line)
                if 'image' in rec:
                    processed.add(str(rec['image']))
            except json.JSONDecodeError:
                continue
    
    # Get image directory for this category
    image_dir = Path(f"amazon_data_bench/{category_name}/images")
    if not image_dir.exists():
        logging.warning(f"Image directory {image_dir} does not exist for category {category_name}")
        return
    
    # Get all images
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    all_images = sorted([p for ext in exts for p in image_dir.glob(ext)])
    
    # Limit to max_items
    images = all_images[:max_items]
    
    logging.info(f"Category {category_name}: Processing {len(images)} images for titles (max {max_items})")
    
    for img_path in tqdm(images, desc=f"Processing {category_name} images on GPU {gpu_id}"):
        item_id = img_path.stem.split('.')[0]
        if item_id in processed:
            logging.info(f"Skipping already processed {item_id}")
            continue
        
        logging.info(f"Processing {img_path.name}")
        try:
            parsed, raw_output, session = infer_product_title(processor, model, img_path, model_name=model_name)
            parsed['raw_output'] = raw_output
        except Exception as e:
            parsed = {
                'image': img_path.stem.split('.')[0],
                'title': 'error',
                'raw_output': f"Fatal error: {str(e)}"
            }
        
        with open(out_file, 'a', encoding='utf-8') as fout:
            fout.write(json.dumps(parsed, ensure_ascii=False) + '\n')
    
    logging.info(f"Category {category_name}: Completed processing titles on GPU {gpu_id}")

def process_category_descriptions(category_name: str, model_name: str, gpu_id: int, max_items: int = 1000, mode: str = "prompt"):
    """Process descriptions for a specific category to generate image prompts"""
    processor, model = load_model(model_name, gpu_id)
    OUTPUT_DIR = get_output_dir(model_name, mode)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Category-specific output file
    out_file = OUTPUT_DIR / f'{category_name}_prompts_gpu{gpu_id}.jsonl'
    
    # Check existing processed items
    processed = set()
    if out_file.exists():
        for line in out_file.read_text(encoding='utf-8').splitlines():
            try:
                rec = json.loads(line)
                if 'image' in rec:
                    processed.add(str(rec['image']))
            except json.JSONDecodeError:
                continue

    # Read descriptions from the category data file
    data_file = Path(f"amazon_data_bench/{category_name}/{category_name}_data.txt")
    if not data_file.exists():
        logging.warning(f"Data file {data_file} does not exist for category {category_name}")
        return

    # Read all descriptions
    descriptions = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                descriptions.append((parts[0], parts[1]))  # Store both item_id and title
    
    # Limit to max_items
    descriptions = descriptions[:max_items]
    
    logging.info(f"Category {category_name}: Processing {len(descriptions)} descriptions for prompts (max {max_items})")
    
    for item_id, title in tqdm(descriptions, desc=f"Processing {category_name} descriptions on GPU {gpu_id}"):
        if item_id in processed:
            logging.info(f"Skipping already processed item {item_id}")
            continue
        
        logging.info(f"Processing item {item_id}: {title[:100]}...")
        try:
            parsed, raw_output, session = infer_image_prompt(processor, model, title, model_name=model_name)
            parsed['image'] = item_id
            parsed['raw_output'] = raw_output
        except Exception as e:
            parsed = {
                'image': item_id,
                'prompt': 'error',
                'raw_output': f"Fatal error: {str(e)}"
            }
        
        with open(out_file, 'a', encoding='utf-8') as fout:
            fout.write(json.dumps(parsed, ensure_ascii=False) + '\n')
    
    logging.info(f"Category {category_name}: Completed processing prompts on GPU {gpu_id}")

def main():
    parser = argparse.ArgumentParser(description='Amazon Product Processing')
    parser.add_argument('--mode', type=str, required=True, choices=['title', 'prompt', 'title_rec', 'prompt_rec', 'title_bench', 'prompt_bench'],
                      help='Processing mode: title for generating titles from images, prompt for generating image prompts from descriptions, title_rec for rec dataset titles, prompt_rec for rec dataset prompts, title_bench for bench dataset titles, prompt_bench for bench dataset prompts')
    parser.add_argument(
        '--model_name', type=str,
        default='Qwen/Qwen2.5-VL-32B-Instruct',
        help='HuggingFace model name'
    )
    parser.add_argument('--gpu_id', type=int, required=True, help='GPU ID to use')
    parser.add_argument('--category', type=str, help='Specific category to process (optional)')
    parser.add_argument('--max_items', type=int, default=1000, help='Maximum items to process per category')
    args = parser.parse_args()
    
    categories = get_available_categories()
    
    if args.category:
        if args.category not in categories:
            logging.error(f"Category {args.category} not found. Available categories: {categories}")
            return
        categories = [args.category]
    
    logging.info(f"Processing {len(categories)} categories with model {args.model_name} on GPU {args.gpu_id}")
    logging.info(f"Mode: {args.mode}")
    
    for category in categories:
        logging.info(f"Processing category: {category}")
        try:
            if args.mode == "title" or args.mode == "title_rec" or args.mode == "title_bench":
                process_category_images(category, args.model_name, args.gpu_id, args.max_items, args.mode)
            else:  # prompt or prompt_rec mode
                process_category_descriptions(category, args.model_name, args.gpu_id, args.max_items, args.mode)
        except Exception as e:
            logging.error(f"Failed to process category {category}: {e}")
            continue
    
    logging.info("All categories processed!")

if __name__ == '__main__':
    main()

'''
# Examples for running the script:

# Generate titles from images for all categories:
python imagine.py --mode title --model_name Qwen/Qwen2.5-VL-32B-Instruct --gpu_id 0

# Generate image prompts from descriptions for all categories:
python imagine.py --mode prompt --model_name Qwen/Qwen2.5-VL-32B-Instruct --gpu_id 0

# Generate titles for specific category:
python imagine.py --mode title --model_name Qwen/Qwen2.5-VL-32B-Instruct --gpu_id 0 --category All_Beauty

# Generate prompts for specific category:
python imagine.py --mode prompt --model_name google/gemma-3-4b-it --gpu_id 1 --category Electronics

# Process with custom max items:
python imagine.py --mode title --model_name Qwen/Qwen2.5-VL-32B-Instruct --gpu_id 0 --max_items 500

# Using InternVL3 models:
# Generate titles with InternVL3-8B:
python imagine.py --mode title --model_name OpenGVLab/InternVL3-8B-Instruct --gpu_id 0

# Generate prompts with InternVL3-78B:
python imagine.py --mode prompt --model_name OpenGVLab/InternVL3-78B-Instruct --gpu_id 1

# Generate titles for specific category with InternVL3-8B:
python imagine.py --mode title --model_name OpenGVLab/InternVL3-8B-Instruct --gpu_id 0 --category Electronics

# Generate prompts for specific category with InternVL3-78B:
python imagine.py --mode prompt --model_name OpenGVLab/InternVL3-78B-Instruct --gpu_id 1 --category All_Beauty

# Using InternVL3 models with lmdeploy backend (faster inference):
# Generate titles with InternVL3-8B using lmdeploy:
python imagine.py --mode title --model_name OpenGVLab/InternVL3-8B-Instruct --gpu_id 0

# Generate prompts with InternVL3-78B using lmdeploy:
python imagine.py --mode prompt --model_name OpenGVLab/InternVL3-78B-Instruct --gpu_id 1

# Generate titles for specific category with InternVL3-8B using lmdeploy:
python imagine.py --mode title --model_name OpenGVLab/InternVL3-8B-Instruct --gpu_id 0 --category Electronics

# Using LLaVA models:
# Generate titles with LLaVA-1.6-Mistral-7B:
python imagine.py --mode title --model_name llava-hf/llava-v1.6-mistral-7b-hf --gpu_id 0

# Generate prompts with LLaVA-1.6-34B:
python imagine.py --mode prompt --model_name llava-hf/llava-v1.6-34b-hf --gpu_id 1

# Generate titles for specific category with LLaVA-1.6-Mistral-7B:
python imagine.py --mode title --model_name llava-hf/llava-v1.6-mistral-7b-hf --gpu_id 0 --category Electronics

# Generate prompts for specific category with LLaVA-1.6-34B:
python imagine.py --mode prompt --model_name llava-hf/llava-v1.6-34b-hf --gpu_id 1 --category All_Beauty

# Process with custom max items using LLaVA-1.6-34B:
python imagine.py --mode title --model_name llava-hf/llava-v1.6-34b-hf --gpu_id 0 --max_items 500
'''