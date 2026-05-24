#!/usr/bin/env python3
"""
Script to generate images using stabilityai/stable-diffusion-3.5-large-turbo model
from prompts in organized_outputs directory
"""
import os
import pandas as pd
import logging
from tqdm import tqdm
import torch
from diffusers import AutoPipelineForText2Image
from pathlib import Path
import glob
import argparse
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_model(model_name="stabilityai/stable-diffusion-3.5-large-turbo"):
    """Load the specified model"""
    logging.info(f"Loading {model_name} model...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipe.to("cuda")
    return pipe

def get_image_generation_model():
    """Get the image generation model (always StabilityAI)"""
    return "stabilityai/stable-diffusion-3.5-large-turbo"

def process_prompts_file(prompts_file, pipe, output_base_dir, max_sequence_length=512):
    """Process a single prompts file and generate images"""
    logging.info(f"Processing prompts file: {prompts_file}")
    
    # Determine model and category from file path
    file_path = Path(prompts_file)
    prompt_model_name = file_path.parts[-3]  # organized_outputs/model_name/prompts/file.csv
    category = file_path.stem.split('_gpu')[0]  # Extract category name
    
    # Create output directory for this model and category
    output_dir = output_base_dir / prompt_model_name / category
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Prompt Model: {prompt_model_name}, Category: {category}")
    logging.info(f"Using max_sequence_length: {max_sequence_length}")
    
    # Read the prompts file
    try:
        df = pd.read_csv(prompts_file)
        logging.info(f"Loaded {len(df)} prompts from {prompts_file}")
    except Exception as e:
        logging.error(f"Failed to read prompts file {prompts_file}: {e}")
        return
    
    total_entries = len(df)
    successful_generations = 0
    failed_generations = 0
    skipped_generations = 0
    long_prompts = 0
    
    for idx, row in tqdm(df.iterrows(), total=total_entries, desc=f"Generating {category} images"):
        item_id = str(row['item_id'])
        prompt = str(row['prompt'])
        
        if not item_id or not prompt or prompt.lower() == 'nan':
            logging.warning(f"Missing id or prompt in row {idx}")
            failed_generations += 1
            continue
        
        # Check if image already exists
        output_path = output_dir / f"{item_id}.jpg"
        if output_path.exists():
            logging.debug(f"Image for {item_id} already exists, skipping...")
            skipped_generations += 1
            continue
        
        # Track long prompts
        if len(prompt) > 300:  # Rough estimate for long prompts
            long_prompts += 1
            logging.debug(f"Processing long prompt for {item_id} ({len(prompt)} characters)")
            
        try:
            # Generate image with specified max_sequence_length
            image = pipe(
                prompt=prompt,
                num_inference_steps=4,
                guidance_scale=0.0,
                max_sequence_length=max_sequence_length,
            ).images[0]

            # Save the generated image
            image.save(output_path)
            successful_generations += 1
            logging.debug(f"Generated image for {item_id}")
            
        except Exception as e:
            logging.error(f"Failed to generate image for {item_id}: {str(e)}")
            failed_generations += 1
    
    # Log statistics for this file
    logging.info(f"=== {category} ({prompt_model_name}) Statistics ===")
    logging.info(f"Total entries: {total_entries}")
    logging.info(f"Successfully generated: {successful_generations}")
    logging.info(f"Skipped (already existed): {skipped_generations}")
    logging.info(f"Failed: {failed_generations}")
    logging.info(f"Long prompts processed: {long_prompts}")
    if total_entries > 0:
        logging.info(f"Success rate: {successful_generations / total_entries * 100:.2f}%")
    
    return {
        'model': prompt_model_name,
        'category': category,
        'total': total_entries,
        'successful': successful_generations,
        'skipped': skipped_generations,
        'failed': failed_generations,
        'long_prompts': long_prompts
    }

def check_model_already_processed(model_name, output_base_dir, organized_dir, categories_to_check=None):
    """Check if a model has already been processed by looking for existing output folders"""
    model_output_dir = output_base_dir / model_name
    
    # If the model output directory doesn't exist, it hasn't been processed
    if not model_output_dir.exists():
        return False
    
    # Check if there are any subdirectories (categories) in the model output directory
    category_dirs = [d for d in model_output_dir.iterdir() if d.is_dir()]
    
    if not category_dirs:
        return False
    
    # If specific categories are provided, only check those
    if categories_to_check:
        processed_categories = []
        for category in categories_to_check:
            category_dir = model_output_dir / category
            if category_dir.exists():
                image_files = list(category_dir.glob("*.jpg"))
                if image_files:
                    logging.info(f"Model {model_name} already has generated images in {category_dir}")
                    processed_categories.append(category)
        
        # Return True if all specified categories are already processed
        if processed_categories:
            logging.info(f"Already processed categories for {model_name}: {processed_categories}")
            return processed_categories
        return False
    
    # If no specific categories, check if any category has generated images
    for category_dir in category_dirs:
        image_files = list(category_dir.glob("*.jpg"))
        if image_files:
            logging.info(f"Model {model_name} already has generated images in {category_dir}")
            return True
    
    return False

def get_available_categories(organized_dir, model_name):
    """Get available categories for a specific model"""
    model_dir = organized_dir / model_name / "prompts"
    if not model_dir.exists():
        return []
    
    categories = set()
    for file_path in model_dir.glob("*.csv"):
        category = file_path.stem.split('_gpu')[0]
        categories.add(category)
    
    return sorted(list(categories))

def main():
    """Main function to process all prompts files"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate images from prompts using specified model')
    parser.add_argument('--model', type=str, default="stabilityai/stable-diffusion-3.5-large",
                       help='Model name to use (e.g., stabilityai/stable-diffusion-3.5-large, gemma_vl_4b, qwen_vl_7b)')
    parser.add_argument('--models', type=str, nargs='+',
                       help='Multiple model names to process (e.g., gemma_vl_4b qwen_vl_7b)')
    parser.add_argument('--categories', type=str, nargs='+',
                       help='Specific categories to process (e.g., All_Beauty Toys_and_Games Video_Games Home_and_Kitchen)')
    parser.add_argument('--output-dir', type=str, default='./generated_images_bench',
                       help='Base output directory for generated images')
    parser.add_argument('--organized-dir', type=str, default='./organized_outputs_bench',
                       help='Directory containing organized prompts files')
    parser.add_argument('--force', action='store_true',
                       help='Force processing even if model folder already exists')
    parser.add_argument('--list-categories', action='store_true',
                       help='List available categories for specified models and exit')
    parser.add_argument('--max-sequence-length', type=int, default=512,
                       help='Maximum sequence length for text processing (default: 512)')
    
    args = parser.parse_args()
    
    # Determine which models to process
    if args.models:
        models_to_process = args.models
        logging.info(f"Processing multiple models: {models_to_process}")
    else:
        models_to_process = [args.model]
        logging.info(f"Processing single model: {args.model}")
    
    # Create base output directory
    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(exist_ok=True)
    
    # Find all prompts CSV files in organized_outputs
    organized_dir = Path(args.organized_dir)
    if not organized_dir.exists():
        logging.error(f"{args.organized_dir} directory not found!")
        return
    
    # If --list-categories is specified, show available categories and exit
    if args.list_categories:
        logging.info("Available categories for each model:")
        for model_name in models_to_process:
            categories = get_available_categories(organized_dir, model_name)
            logging.info(f"{model_name}: {categories}")
        return
    
    # Load the image generation model (always StabilityAI)
    image_gen_model = get_image_generation_model()
    logging.info(f"Using image generation model: {image_gen_model}")
    logging.info(f"Using max_sequence_length: {args.max_sequence_length}")
    
    try:
        pipe = load_model(image_gen_model)
    except Exception as e:
        logging.error(f"Failed to load image generation model {image_gen_model}: {e}")
        return
    
    # Process each model
    all_stats = []
    
    for model_name in models_to_process:
        logging.info("=" * 60)
        logging.info(f"🔄 Processing prompts from model: {model_name}")
        logging.info("=" * 60)
        
        # Get available categories for this model
        available_categories = get_available_categories(organized_dir, model_name)
        
        # Check if model directory exists
        model_specific_dir = organized_dir / model_name / "prompts"
        if not model_specific_dir.exists():
            logging.warning(f"No prompts directory found for model {model_name}")
            continue
        
        # Filter categories if specified
        if args.categories:
            categories_to_process = [cat for cat in args.categories if cat in available_categories]
            if not categories_to_process:
                logging.warning(f"None of the specified categories {args.categories} found for model {model_name}")
                continue
            logging.info(f"Processing categories: {categories_to_process}")
        else:
            categories_to_process = available_categories
            logging.info(f"Processing all available categories: {categories_to_process}")
        
        # Check if model has already been processed for these categories
        processed_categories = check_model_already_processed(model_name, output_base_dir, organized_dir, categories_to_process)
        if not args.force and processed_categories:
            # Filter out already processed categories
            remaining_categories = [cat for cat in categories_to_process if cat not in processed_categories]
            if not remaining_categories:
                logging.info(f"⏭️  Skipping {model_name} - all specified categories already processed. Use --force to reprocess.")
                continue
            else:
                logging.info(f"🔄 Processing {model_name} - skipping already processed categories: {processed_categories}")
                logging.info(f"🔄 Remaining categories to process: {remaining_categories}")
                categories_to_process = remaining_categories
        
        # Process each category for this model
        for category in categories_to_process:
            # Find the prompts file for this category - look for any gpu*.csv file
            prompts_file = None
            for gpu_file in model_specific_dir.glob(f"{category}_gpu*.csv"):
                prompts_file = gpu_file
                break
            
            if not prompts_file:
                logging.warning(f"Prompts file not found for category {category} in model {model_name}")
                continue
            
            try:
                stats = process_prompts_file(prompts_file, pipe, output_base_dir, args.max_sequence_length)
                if stats:
                    # Update stats to reflect the current model being used
                    stats['model'] = model_name
                    all_stats.append(stats)
            except Exception as e:
                logging.error(f"Failed to process {prompts_file}: {e}")
    
    # Print final summary
    logging.info("=" * 60)
    logging.info("📊 FINAL SUMMARY")
    logging.info("=" * 60)
    
    total_successful = 0
    total_skipped = 0
    total_failed = 0
    total_entries = 0
    total_long_prompts = 0
    
    for stats in all_stats:
        logging.info(f"{stats['model']}/{stats['category']}: "
                    f"{stats['successful']} generated, "
                    f"{stats['skipped']} skipped, "
                    f"{stats['failed']} failed, "
                    f"{stats.get('long_prompts', 0)} long prompts")
        total_successful += stats['successful']
        total_skipped += stats['skipped']
        total_failed += stats['failed']
        total_entries += stats['total']
        total_long_prompts += stats.get('long_prompts', 0)
    
    logging.info("=" * 60)
    logging.info(f"TOTAL: {total_successful} generated, {total_skipped} skipped, {total_failed} failed, {total_long_prompts} long prompts")
    if total_entries > 0:
        logging.info(f"Overall success rate: {total_successful / total_entries * 100:.2f}%")
    logging.info(f"Images saved in: {output_base_dir}")

if __name__ == '__main__':
    main() 



'''
# Basic usage with default settings:
CUDA_VISIBLE_DEVICES=0 python generate_images_from_prompts.py --models "gemma_vl_4b" "qwen_vl_7b" --categories "All_Beauty" "Toys_and_Games" "Video_Games" "Home_and_Kitchen"

# Usage with longer sequence length to handle very long prompts:
CUDA_VISIBLE_DEVICES=0 python generate_images_from_prompts.py \
    --models "gemma_vl_4b" "qwen_vl_7b" \
    --categories "All_Beauty" "Toys_and_Games" "Video_Games" "Home_and_Kitchen" \
    --max-sequence-length 1024

# To list available categories:
CUDA_VISIBLE_DEVICES=0 python generate_images_from_prompts.py --models "gemma_vl_4b" "qwen_vl_7b" --list-categories

# Force reprocessing with custom sequence length:
CUDA_VISIBLE_DEVICES=0 python generate_images_from_prompts.py \
    --models "gemma_vl_4b" "qwen_vl_7b" \
    --categories "All_Beauty" "Toys_and_Games" "Video_Games" "Home_and_Kitchen" \
    --max-sequence-length 2048 \
    --force


CUDA_VISIBLE_DEVICES=0 python generate_images_from_prompts.py \
    --models "gemma_vl_4b" "qwen_vl_7b" \
    --categories "All_Beauty" "Toys_and_Games" "Video_Games" "Home_and_Kitchen"

CUDA_VISIBLE_DEVICES=0 python generate_images_from_prompts.py     --models "gemma_vl_4b" "qwen_vl_72b"     --categories "All_Beauty" "Toys_and_Games" "Video_Games" "Home_and_Kitchen" "Electronics" "Industrial_and_Scientific" "Office_Products" "Musical_Instruments" "Arts_Crafts_and_Sewing"
CUDA_VISIBLE_DEVICES=1 python generate_images_from_prompts.py     --models "gemma_vl_12b" "qwen_vl_32b"     --categories "All_Beauty" "Toys_and_Games" "Video_Games" "Home_and_Kitchen" "Electronics" "Industrial_and_Scientific" "Office_Products" "Musical_Instruments" "Arts_Crafts_and_Sewing"
CUDA_VISIBLE_DEVICES=2 python generate_images_from_prompts.py     --models "gemma_vl_27b" "qwen_vl_72b"     --categories "All_Beauty" "Toys_and_Games" "Video_Games" "Home_and_Kitchen" "Electronics" "Industrial_and_Scientific" "Office_Products" "Musical_Instruments" "Arts_Crafts_and_Sewing"
CUDA_VISIBLE_DEVICES=2 python generate_images_from_prompts.py     --models "qwen_vl_unknown"     --categories "All_Beauty" "Toys_and_Games" "Video_Games" "Home_and_Kitchen" "Electronics" "Industrial_and_Scientific" "Office_Products" "Musical_Instruments" "Arts_Crafts_and_Sewing"

#
CUDA_VISIBLE_DEVICES=1 python generate_images_from_prompts.py     --models "qwen_vl_7b" "qwen_vl_32b"   --categories "All_Beauty" "Toys_and_Games" "Video_Games" "Home_and_Kitchen" "Electronics" "Industrial_and_Scientific" "Office_Products" "Musical_Instruments" "Arts_Crafts_and_Sewing"
CUDA_VISIBLE_DEVICES=2 python generate_images_from_prompts.py     --models "gemma_vl_4b" "gemma_vl_12b"  --categories "All_Beauty" "Toys_and_Games" "Video_Games" "Home_and_Kitchen" "Electronics" "Industrial_and_Scientific" "Office_Products" "Musical_Instruments" "Arts_Crafts_and_Sewing"
CUDA_VISIBLE_DEVICES=3 python generate_images_from_prompts.py     --models "qwen_vl_72b"  --categories "All_Beauty" "Toys_and_Games" "Video_Games" "Home_and_Kitchen" "Electronics" "Industrial_and_Scientific" "Office_Products" "Musical_Instruments" "Arts_Crafts_and_Sewing"
CUDA_VISIBLE_DEVICES=0 python generate_images_from_prompts.py     --models "gemma_vl_27b" --categories "All_Beauty" "Toys_and_Games" "Video_Games" "Home_and_Kitchen" "Electronics"
CUDA_VISIBLE_DEVICES=3 python generate_images_from_prompts.py     --models "gemma_vl_27b" --categories  "Industrial_and_Scientific" "Office_Products" "Musical_Instruments" "Arts_Crafts_and_Sewing"


CUDA_VISIBLE_DEVICES=1 python generate_images_from_prompts.py     --models "qwen_vl_7b" --categories  "Toys_and_Games"

'''