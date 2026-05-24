#!/usr/bin/env python3
"""
Extract title and image features for Amazon product data with missing modality completion.

This script supports:
• Missing rate setting with random_state=42 for reproducible missing item selection
• MLLM-generated text/image completion for missing modalities
• Comparison experiments with zeros and random vectors
• Multi-model support for all available MLLMs

Expected directory layout:
    MLLM Text: /data/junch/LLM_IMG_MISS_Dataset/organized_outputs_rec/{model}/titles/{category}_gpu*.csv
    MLLM Images: /data/junch/LLM_IMG_MISS_Dataset/generated_images_rec/{model}/{category}/
    Original Data: /data/junch/LLM_IMG_MISS_Dataset/amazon_data_rec/{category}/

Outputs are stored in ``output_dir/{category}/`` as:
    - text_feat_missing_{rate}_{model}.npy      (N, d_text)
    - image_feat_missing_{rate}_{model}.npy     (N, d_img)
    - text_feat_missing_{rate}_zeros.npy        (N, d_text) - comparison
    - image_feat_missing_{rate}_zeros.npy       (N, d_img) - comparison
    - text_feat_missing_{rate}_random.npy       (N, d_text) - comparison
    - image_feat_missing_{rate}_random.npy      (N, d_img) - comparison
    - item_ids.json                             list[str] itemIDs in the same order as features
    - missing_ids_{rate}.json                   list[str] missing item IDs for this rate

Usage
-----
# Process single category with specific model
python extract_features_missing.py --category All_Beauty \
                                   --model qwen_vl_7b \
                                   --base_dir /data/junch/LLM_IMG_MISS_Dataset/amazon_data_rec \
                                   --output_dir ./feature_cache

# Process all categories with all models
python extract_features_missing.py --all_categories_models \
                                   --base_dir /data/junch/LLM_IMG_MISS_Dataset/amazon_data_rec \
                                   --output_dir ./feature_cache
"""

import os
import json
import argparse
import re
import random
from pathlib import Path
from typing import List, Tuple, Dict, Set

import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
import timm
from torchvision import transforms

# -----------------------------------------------------------------------------
# Constants and Configuration
# -----------------------------------------------------------------------------

# Base directories
ORIGINAL_DATA_DIR = Path("/data/junch/LLM_IMG_MISS_Dataset/amazon_data_rec")
MLLM_TEXT_DIR = Path("/data/junch/LLM_IMG_MISS_Dataset/organized_outputs_rec")
MLLM_IMAGE_DIR = Path("/data/junch/LLM_IMG_MISS_Dataset/generated_images_rec")
MMREC_DATA_DIR = Path("/data/junch/MMRec_Missing/data")

# Random seed for reproducible missing item selection
RANDOM_SEED = 42

# Fixed missing rates to generate
MISSING_RATES = [0.2, 0.5, 0.8, 1.0]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def get_available_models() -> List[str]:
    """Get list of available MLLM models from organized_outputs_rec directory."""
    if not MLLM_TEXT_DIR.exists():
        raise RuntimeError(f"MLLM text directory not found: {MLLM_TEXT_DIR}")
    
    models = [d.name for d in MLLM_TEXT_DIR.iterdir() if d.is_dir()]
    print(f"Available models: {models}")
    return models

def get_model_text_file(model: str, category: str) -> Path:
    """Get the text file path for a specific model and category."""
    model_dir = MLLM_TEXT_DIR / model / "titles"
    if not model_dir.exists():
        raise RuntimeError(f"Model text directory not found: {model_dir}")
    
    # Find the CSV file for this category (remove _Small and _gpu* suffix)
    base_category = category.replace('_Small', '')
    base_category = re.sub(r'_gpu\d+$', '', base_category)
    pattern = f"{base_category}_gpu*.csv"
    
    matching_files = list(model_dir.glob(pattern))
    if not matching_files:
        raise RuntimeError(f"No text file found for {model}/{category} in {model_dir}")
    
    # Use the first matching file (usually the largest one)
    return matching_files[0]

def get_model_image_dir(model: str, category: str) -> Path:
    """Get the image directory path for a specific model and category."""
    # Use base category name (without _Small) for image directory lookup
    base_category = category.replace('_Small', '')
    image_dir = MLLM_IMAGE_DIR / model / base_category
    if not image_dir.exists():
        raise RuntimeError(f"Model image directory not found: {image_dir}")
    return image_dir

def load_missing_ids(category: str, missing_rate: float, output_dir: Path) -> Set[str]:
    """Load or generate missing item IDs for a category and missing rate."""
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    missing_file = output_dir / f"missing_ids_{missing_rate:.1f}.json"
    
    if missing_file.exists():
        print(f"Loading existing missing IDs from {missing_file}")
        with open(missing_file, 'r') as f:
            return set(json.load(f))
    
    # Generate missing IDs
    print(f"Generating missing IDs for rate {missing_rate}")
    
    # Load item mapping from MMRec data directory
    # Use original category name (with _Small) for MMRec data directory lookup
    category_lower = category.lower()
    item_map_path = MMREC_DATA_DIR / category_lower / 'i_id_mapping.csv'
    if not item_map_path.exists():
        raise RuntimeError(f"Item mapping not found: {item_map_path}. Please run process_all.py first.")
    
    item_map_df = pd.read_csv(item_map_path)
    all_item_ids = set(item_map_df['original_id'].tolist())
    
    # Set random seed for reproducible selection
    random.seed(RANDOM_SEED)
    
    # Select missing items
    num_missing = int(len(all_item_ids) * missing_rate)
    missing_ids = set(random.sample(list(all_item_ids), num_missing))
    
    # Save missing IDs
    with open(missing_file, 'w') as f:
        json.dump(list(missing_ids), f)
    
    print(f"Generated {len(missing_ids)} missing IDs out of {len(all_item_ids)} total items")
    return missing_ids

def read_category_txt(txt_path: Path) -> Tuple[List[str], List[str]]:
    """Read <asin> and <title> from a *_data.txt file."""
    item_ids, titles = [], []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue  # skip malformed lines
            item_ids.append(parts[0])
            titles.append(parts[1])
    return item_ids, titles

def load_mllm_text_data(model: str, category: str) -> Dict[str, str]:
    """Load MLLM-generated text data for a model and category."""
    text_file = get_model_text_file(model, category)
    print(f"Loading MLLM text from: {text_file}")
    
    df = pd.read_csv(text_file)
    if 'item_id' not in df.columns or 'title' not in df.columns:
        raise RuntimeError(f"Invalid CSV format in {text_file}. Expected columns: item_id, title")
    
    # Create mapping from item_id to generated title
    text_map = dict(zip(df['item_id'], df['title']))
    print(f"Loaded {len(text_map)} MLLM-generated titles")
    return text_map

def encode_titles(titles: List[str], model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 512, device: str = 'cpu') -> np.ndarray:
    """Encode titles with SentenceTransformer."""
    model = SentenceTransformer(model_name, device=device)
    emb_list = []
    for i in tqdm(range(0, len(titles), batch_size), desc='Encoding titles'):
        batch = titles[i:i + batch_size]
        emb = model.encode(batch, convert_to_numpy=True)
        emb_list.append(emb)
    return np.concatenate(emb_list, axis=0)

def prepare_vit(device: str):
    """Load ViT-Base-Patch16-224 model from timm for feature extraction."""
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.eval()
    model.to(device)

    # Define preprocessing transforms similar to timm default
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    return model, preprocess

def load_mllm_image_data(model: str, category: str) -> Set[str]:
    """Load MLLM-generated image data for a model and category."""
    image_dir = get_model_image_dir(model, category)
    print(f"Loading MLLM images from: {image_dir}")
    
    # Get all available image files
    available_images = set()
    for ext in ('.jpg', '.png', '.jpeg'):
        available_images.update([f.stem for f in image_dir.glob(f"*{ext}")])
    
    print(f"Found {len(available_images)} MLLM-generated images")
    return available_images

def encode_images_with_mllm(item_ids: List[str], original_images_dir: Path, 
                           mllm_images_dir: Path, missing_ids: Set[str], 
                           device: str, batch_size: int = 64) -> np.ndarray:
    """Encode product images with ViT-base. Use MLLM-generated images for missing items."""
    model, preprocess = prepare_vit(device)

    d_img = model.num_features if hasattr(model, 'num_features') else 768

    embs = np.zeros((len(item_ids), d_img), dtype=np.float32)
    present_flags = np.zeros(len(item_ids), dtype=bool)

    def load_image(item_id: str, use_mllm: bool = False):
        """Load image from either original or MLLM directory."""
        if use_mllm:
            # Try MLLM-generated image first
            for ext in ('.jpg', '.png', '.jpeg'):
                p = mllm_images_dir / f"{item_id}{ext}"
                if p.exists():
                    return Image.open(p).convert('RGB')
        else:
            # Try original image
            for ext in ('.jpg', '.png', '.jpeg'):
                p = original_images_dir / f"{item_id}{ext}"
                if p.exists():
                    return Image.open(p).convert('RGB')
        return None

    img_batch, idx_batch = [], []

    for idx, item_id in enumerate(tqdm(item_ids, desc='Gathering images')):
        # Determine which image to use
        is_missing = item_id in missing_ids
        img = None
        
        if is_missing:
            # For missing items, try MLLM-generated image first, then fall back to original
            img = load_image(item_id, use_mllm=True)
            if img is None:
                img = load_image(item_id, use_mllm=False)
        else:
            # For non-missing items, use original image
            img = load_image(item_id, use_mllm=False)
        
        if img is None:
            continue
            
        img_batch.append(preprocess(img))
        idx_batch.append(idx)
        
        if len(img_batch) == batch_size or idx == len(item_ids) - 1:
            if img_batch:
                with torch.no_grad():
                    imgs_tensor = torch.stack(img_batch).to(device)
                    feat_tok = model.forward_features(imgs_tensor)  # (B, T, C)
                    # Take CLS token (index 0)
                    feat = feat_tok[:, 0, :].cpu().numpy()
                embs[idx_batch] = feat
                present_flags[idx_batch] = True
            img_batch, idx_batch = [], []

    if present_flags.any():
        avg_emb = embs[present_flags].mean(axis=0)
        embs[~present_flags] = avg_emb
    else:
        raise RuntimeError('No images found for any item.')

    return embs

# -----------------------------------------------------------------------------
# Main Processing Functions
# -----------------------------------------------------------------------------

def process_category_with_missing(category: str, model: str, 
                                base_dir: Path, output_dir: Path, device: str):
    """Process a category with missing modality completion for all missing rates."""
    print(f"\n=== Processing {category} with {model} (all missing rates: {MISSING_RATES}) ===")
    
    # Handle _small suffix by removing it for data file lookup
    base_category_name = category.replace('_Small', '')
    cat_dir = base_dir / base_category_name
    txt_path = cat_dir / f"{base_category_name}_data.txt"
    images_dir = cat_dir / 'images'

    # Check required files exist
    if not txt_path.exists():
        print(f"[Skip] txt file not found: {txt_path}")
        return
    if not images_dir.exists():
        print(f"[Skip] images directory not found: {images_dir}")
        return

    # Load all original titles and map them to item IDs
    all_item_ids, all_titles = read_category_txt(txt_path)
    original_title_map = dict(zip(all_item_ids, all_titles))

    # Load MLLM-generated text data
    try:
        mllm_text_map = load_mllm_text_data(model, category)
    except Exception as e:
        print(f"[Skip] Failed to load MLLM text data: {e}")
        return

    # Load MLLM-generated image data
    try:
        mllm_image_set = load_mllm_image_data(model, category)
    except Exception as e:
        print(f"[Skip] Failed to load MLLM image data: {e}")
        return

    # Load filtered item IDs from process_all.py
    # Use original category name (with _Small) for MMRec data directory lookup
    category_lower = category.lower()
    item_map_path = MMREC_DATA_DIR / category_lower / 'i_id_mapping.csv'
    if not item_map_path.exists():
        print(f"[Skip] Item mapping not found: {item_map_path}")
        print(f"      Please run process_all.py for '{category}' first.")
        return

    item_map_df = pd.read_csv(item_map_path)
    item_map_df = item_map_df.sort_values('new_id')
    filtered_item_ids = item_map_df['original_id'].tolist()
    
    print(f"Total items in i_id_mapping.csv: {len(filtered_item_ids)}")

    # Prepare final lists of item IDs and titles for feature extraction
    # We need to ensure the final features match exactly with filtered_item_ids
    item_ids = []
    titles = []
    valid_indices = []  # Track which indices in filtered_item_ids have valid data
    
    for idx, pid in enumerate(filtered_item_ids):
        if pid in original_title_map:
            item_ids.append(pid)
            titles.append(original_title_map[pid])
            valid_indices.append(idx)
        else:
            print(f"[Warning] Item {pid} not found in original title map")

    print(f"Found titles for {len(item_ids)} out of {len(filtered_item_ids)} filtered items")
    
    if len(item_ids) != len(filtered_item_ids):
        print(f"[Warning] Mismatch: Found titles for {len(item_ids)} of {len(filtered_item_ids)} filtered items.")
        print("This will result in features with different dimensions than expected.")

    # Count how many missing items have MLLM-generated images
    print(f"Missing items with MLLM images: {len([pid for pid in item_ids if pid in mllm_image_set])}/{len(item_ids)}")

    # Encode titles once (will be reused for all missing rates)
    print("Encoding titles...")
    title_feat = encode_titles(titles, device=device)
    print(f"Title features shape: {title_feat.shape}")

    # Encode images once (will be reused for all missing rates)
    print("Encoding images...")
    image_feat = encode_images_with_mllm(item_ids, images_dir, get_model_image_dir(model, category), set(), device=device)
    print(f"Image features shape: {image_feat.shape}")

    # Get feature dimensions for later use
    d_text = title_feat.shape[1]
    d_img = image_feat.shape[1]
    
    # Ensure features have the correct dimension (matching filtered_item_ids)
    if len(title_feat) != len(filtered_item_ids):
        print(f"Adjusting feature dimensions from {len(title_feat)} to {len(filtered_item_ids)}")
        # Create new arrays with correct dimensions
        adjusted_title_feat = np.zeros((len(filtered_item_ids), d_text), dtype=np.float32)
        adjusted_image_feat = np.zeros((len(filtered_item_ids), d_img), dtype=np.float32)
        
        # Fill in the valid items
        for i, idx in enumerate(valid_indices):
            adjusted_title_feat[idx] = title_feat[i]
            adjusted_image_feat[idx] = image_feat[i]
        
        title_feat = adjusted_title_feat
        image_feat = adjusted_image_feat
        
        print(f"Adjusted title features shape: {title_feat.shape}")
        print(f"Adjusted image features shape: {image_feat.shape}")

    # Pre-encode all MLLM images for missing items (do this once)
    print("Pre-encoding MLLM images for missing items...")
    mllm_image_features = {}
    mllm_images_dir = get_model_image_dir(model, category)
    
    # Get all missing items that have MLLM images
    all_missing_items = set()
    for missing_rate in MISSING_RATES:
        missing_ids = load_missing_ids(category, missing_rate, output_dir)
        all_missing_items.update(missing_ids)
    
    # Encode MLLM images for all missing items at once
    missing_items_with_mllm_images = [item_id for item_id in all_missing_items if item_id in mllm_image_set]
    if missing_items_with_mllm_images:
        print(f"Encoding {len(missing_items_with_mllm_images)} MLLM images...")
        # Encode using MLLM images. Treat every item in this list as "missing" so that
        # encode_images_with_mllm will prioritise the MLLM-generated image over the
        # original image when both exist.
        mllm_missing_set = set(missing_items_with_mllm_images)
        mllm_image_embs = encode_images_with_mllm(
            missing_items_with_mllm_images,
            images_dir,
            mllm_images_dir,
            mllm_missing_set,
            device=device,
        )
        for i, item_id in enumerate(missing_items_with_mllm_images):
            mllm_image_features[item_id] = mllm_image_embs[i]
    
    print(f"Pre-encoded {len(mllm_image_features)} MLLM images")

    # Pre-encode all MLLM titles for missing items (do this once)
    print("Pre-encoding MLLM titles for missing items...")
    mllm_text_features = {}
    
    # Get all missing items that have MLLM titles
    missing_items_with_mllm_titles = [item_id for item_id in all_missing_items if item_id in mllm_text_map]
    if missing_items_with_mllm_titles:
        mllm_titles_to_encode = [mllm_text_map[item_id] for item_id in missing_items_with_mllm_titles]
        print(f"Encoding {len(mllm_titles_to_encode)} MLLM titles...")
        mllm_title_embs = encode_titles(mllm_titles_to_encode, device=device)
        for i, item_id in enumerate(missing_items_with_mllm_titles):
            mllm_text_features[item_id] = mllm_title_embs[i]
    
    print(f"Pre-encoded {len(mllm_text_features)} MLLM titles")

    # Save outputs for all missing rates
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate features for each missing rate
    for missing_rate in MISSING_RATES:
        print(f"\n--- Processing missing rate {missing_rate} ---")
        
        # Load missing IDs for this rate
        missing_ids = load_missing_ids(category, missing_rate, output_dir)
        
        # Create title features for this missing rate
        missing_title_feat = title_feat.copy()
        missing_image_feat = image_feat.copy()
        
        # Replace text features for missing items using pre-encoded MLLM titles
        missing_text_count = 0
        for idx, item_id in enumerate(filtered_item_ids):
            if item_id in missing_ids and item_id in mllm_text_features:
                missing_title_feat[idx] = mllm_text_features[item_id]
                missing_text_count += 1
        
        print(f"Replaced {missing_text_count} missing items with MLLM-generated titles")
        
        # Replace image features for missing items using pre-encoded MLLM images
        missing_image_count = 0
        for idx, item_id in enumerate(filtered_item_ids):
            if item_id in missing_ids and item_id in mllm_image_features:
                missing_image_feat[idx] = mllm_image_features[item_id]
                missing_image_count += 1
        
        print(f"Replaced {missing_image_count} missing items with MLLM-generated images")
        
        # Save MLLM-completed features
        np.save(output_dir / f'text_feat_missing_{missing_rate:.1f}_{model}.npy', missing_title_feat)
        np.save(output_dir / f'image_feat_missing_{missing_rate:.1f}_{model}.npy', missing_image_feat)
        
        # -------------------------------------------------------------
        # Baseline comparisons: zeros and random vectors **only for the
        # items that are missing** at the current missing-rate. This makes the
        # baseline vary with the missing-rate instead of being identical for
        # all rates.
        # -------------------------------------------------------------

        mask_missing = np.array([pid in missing_ids for pid in filtered_item_ids])

        # Zeros comparison – keep original features for observed items, set
        # features for missing items to zeros.
        text_zeros = title_feat.copy()
        image_zeros = image_feat.copy()
        text_zeros[mask_missing] = 0.0
        image_zeros[mask_missing] = 0.0
        np.save(output_dir / f'text_feat_missing_{missing_rate:.1f}_zeros.npy', text_zeros)
        np.save(output_dir / f'image_feat_missing_{missing_rate:.1f}_zeros.npy', image_zeros)

        # Random comparison – replace features **only for missing items** with
        # random values. Use a different seed for each missing-rate so that the
        # random baseline differs across rates while remaining reproducible.
        rng_seed = RANDOM_SEED + int(missing_rate * 100)
        rng = np.random.default_rng(rng_seed)
        text_random = title_feat.copy()
        image_random = image_feat.copy()
        text_random[mask_missing] = rng.standard_normal((mask_missing.sum(), d_text)).astype(np.float32)
        image_random[mask_missing] = rng.standard_normal((mask_missing.sum(), d_img)).astype(np.float32)
        np.save(output_dir / f'text_feat_missing_{missing_rate:.1f}_random.npy', text_random)
        np.save(output_dir / f'image_feat_missing_{missing_rate:.1f}_random.npy', image_random)
        
        print(f"Saved features for missing rate {missing_rate}")
    
    # Generate no-missing features (all original) for comparison
    np.save(output_dir / f'text_feat_no_missing_{model}.npy', title_feat)
    np.save(output_dir / f'image_feat_no_missing_{model}.npy', image_feat)
    
    # Save the ordered list of item IDs corresponding to the features
    with open(output_dir / 'item_ids.json', 'w') as f:
        json.dump(filtered_item_ids, f)

    print(f"Saved features and item IDs to {output_dir}")
    print(f"Generated files for all missing rates: {MISSING_RATES}")
    print(f"Final feature dimensions: {len(filtered_item_ids)} items")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features with missing modality completion')
    parser.add_argument('--base_dir', type=str, default='/data/junch/LLM_IMG_MISS_Dataset/amazon_data_rec', 
                       help='Base directory of dataset')
    parser.add_argument('--output_dir', type=str, default='/data/junch/MMRec_Missing/data', 
                       help='Directory to save features')
    parser.add_argument('--category', type=str, nargs='+', help='Categories to process (e.g., All_Beauty Toys_and_Games)')
    parser.add_argument('--model', type=str, nargs='+', help='MLLM models to use for completion (e.g., qwen_vl_7b qwen_vl_32b)')
    parser.add_argument('--all_categories', action='store_true', 
                       help='Process all categories under base_dir')
    parser.add_argument('--all_categories_models', action='store_true', 
                       help='Process all categories with all available models')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                       help='Computation device')

    args = parser.parse_args()
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)

    if args.all_categories_models:
        # Process all categories with all models
        categories = [p.name for p in base_dir.iterdir() if p.is_dir()]
        models = get_available_models()
        
        for cat in categories:
            for model in models:
                cat_output_dir = output_dir / cat.lower()
                try:
                    process_category_with_missing(cat, model, 
                                               base_dir, cat_output_dir, device=args.device)
                except Exception as e:
                    print(f"Error processing {cat} with {model}: {e}")
                    continue
    elif args.all_categories:
        # Process all categories with specified models
        if args.model is None:
            parser.error('--model is required when using --all_categories')
        
        categories = [p.name for p in base_dir.iterdir() if p.is_dir()]
        models = args.model
        
        for cat in categories:
            for model in models:
                cat_output_dir = output_dir / cat.lower()
                try:
                    process_category_with_missing(cat, model, 
                                               base_dir, cat_output_dir, device=args.device)
                except Exception as e:
                    print(f"Error processing {cat} with {model}: {e}")
                    continue
    else:
        # Process specified categories with specified models
        if args.category is None or args.model is None:
            parser.error('Specify --category and --model or use --all_categories/--all_categories_models')
        
        categories = args.category
        models = args.model
        
        for cat in categories:
            for model in models:
                cat_output_dir = output_dir / cat.lower()
                try:
                    process_category_with_missing(cat, model, 
                                               base_dir, cat_output_dir, device=args.device)
                except Exception as e:
                    print(f"Error processing {cat} with {model}: {e}")
                    continue 

'''
1.
CUDA_VISIBLE_DEVICES=0 python extract_features_missing.py --category Electronics_Small All_Beauty Arts_Crafts_and_Sewing Home_and_Kitchen_Small Industrial_and_Scientific Office_Products Musical_Instruments Toys_and_Games Video_Games \
                                   --model qwen_vl_7b qwen_vl_32b qwen_vl_72b \
                                   --base_dir /data/junch/LLM_IMG_MISS_Dataset/amazon_data_rec 

CUDA_VISIBLE_DEVICES=1 python extract_features_missing.py --category Electronics_Small All_Beauty Arts_Crafts_and_Sewing Home_and_Kitchen_Small Industrial_and_Scientific Office_Products Musical_Instruments Toys_and_Games Video_Games \
                                   --model gemma_vl_4b gemma_vl_12b gemma_vl_27b \
                                   --base_dir /data/junch/LLM_IMG_MISS_Dataset/amazon_data_rec 


'''