#!/usr/bin/env python3
"""
Amazon Review Data Preprocessing Script
Processes Amazon review data from HuggingFace datasets and applies the MMRec preprocessing pipeline.
"""

import os
import json
import pandas as pd
import numpy as np
from collections import Counter
from datasets import load_dataset
import argparse
from pathlib import Path
from tqdm import tqdm

# Base directory containing *_data.txt files with allowed item IDs for each category
WHITELIST_BASE_DIR = Path("/data/junch/LLM_IMG_MISS_Dataset/amazon_data_rec")

def get_amazon_category_mapping():
    """Map local category names to HuggingFace dataset category names"""
    return {
        'All_Beauty': 'All_Beauty',
        'Toys_and_Games': 'Toys_and_Games', 
        'Video_Games': 'Video_Games',
        'Home_and_Kitchen': 'Home_and_Kitchen',
        'Electronics': 'Electronics',
        'Home_and_Kitchen_Small': 'Home_and_Kitchen',
        'Electronics_Small': 'Electronics',
        'Industrial_and_Scientific': 'Industrial_and_Scientific',
        'Office_Products': 'Office_Products',
        'Musical_Instruments': 'Musical_Instruments',
        'Arts_Crafts_and_Sewing': 'Arts_Crafts_and_Sewing'
    }

def load_amazon_reviews(category_name):
    """Load Amazon review data from HuggingFace datasets"""
    category_mapping = get_amazon_category_mapping()
    hf_category = category_mapping.get(category_name, category_name.lower())
    
    try:
        dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{hf_category}", split="full", trust_remote_code=True)
        print(f"Successfully loaded {category_name} dataset with {len(dataset)} reviews")
        return dataset
    except Exception as e:
        print(f"Error loading {category_name} dataset: {e}")
        return None

def convert_to_dataframe(dataset):
    """Convert HuggingFace dataset to pandas DataFrame"""
    # Extract relevant columns from the dataset
    data = []
    for item in tqdm(dataset, desc="Converting to DataFrame"):
        # Assuming the dataset has columns like user_id, item_id, rating, timestamp
        # Adjust these based on actual dataset structure
        data.append({
            'userID': item.get('user_id', item.get('reviewerID', '')),
            # Prefer parent_asin over asin for item identification
            'itemID': item.get('item_id', item.get('parent_asin', item.get('asin', ''))),
            'rating': item.get('rating', item.get('overall', 0)),
            'timestamp': item.get('timestamp', item.get('unixReviewTime', 0))
        })
    
    df = pd.DataFrame(data)
    print(f"Converted to DataFrame with shape: {df.shape}")
    return df

# -----------------------------------------------------------------------------
# Utility: load dataframe with local CSV caching
# -----------------------------------------------------------------------------

def load_or_cache_dataframe(category_name: str, output_dir: str):
    """Load a DataFrame from cached CSV if available; otherwise download and cache.

    The cached file is stored as ``{category}.raw.csv`` inside ``output_dir``.
    If the cache is missing, the function downloads the dataset from HuggingFace,
    converts it to a DataFrame via ``convert_to_dataframe``, saves the CSV, and
    returns the DataFrame.
    """

    # Ensure output directory exists before checking cache
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"{category_name.lower()}-raw.csv")

    if os.path.exists(csv_path):
        print(f"Found cached DataFrame ⇒ loading from {csv_path}")
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded cached DataFrame with shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Warning: failed to read cached CSV (will rebuild). Error: {e}")

    # Cache not present or failed to load: fetch dataset and build DataFrame
    dataset = load_amazon_reviews(category_name)
    if dataset is None:
        raise RuntimeError(f"Failed to load dataset for {category_name}")

    df = convert_to_dataframe(dataset)

    # Save to cache
    try:
        df.to_csv(csv_path, index=False)
        print(f"Cached raw DataFrame to {csv_path}")
    except Exception as e:
        print(f"Warning: failed to write cache CSV. Error: {e}")

    return df

# -----------------------------------------------------------------------------
# Item whitelist filtering
# -----------------------------------------------------------------------------

def filter_df_by_item_whitelist(df: pd.DataFrame, category_name: str):
    """Filter DataFrame to keep only rows whose itemID appears in the category-specific whitelist.

    The whitelist file is expected at::

        /data/junch/LLM_IMG_MISS_Dataset/amazon_data_rec/{Category}/{Category}_data.txt

    with the first column being the *asin* / *parent_asin*.
    
    For categories with _small suffix, it will try to match the base category name.
    """

    # Handle _small suffix by removing it for whitelist lookup
    base_category_name = category_name.replace('_Small', '')
    whitelist_path = WHITELIST_BASE_DIR / base_category_name / f"{base_category_name}_data.txt"

    if not whitelist_path.exists():
        print(f"[Warning] Whitelist file not found for {category_name}: {whitelist_path}. Skipping itemID filtering.")
        return df

    # Read allowed item IDs (first column before tab)
    try:
        allowed_ids = set()
        with open(whitelist_path, 'r') as f:
            for line in f:
                if line.strip():
                    allowed_ids.add(line.split('\t')[0])
    except Exception as e:
        print(f"[Warning] Failed to read whitelist file {whitelist_path}: {e}. Skipping filtering.")
        return df

    before_len = len(df)
    df = df[df['itemID'].isin(allowed_ids)]
    print(f"Filtered with whitelist: {before_len} -> {len(df)} interactions (kept {len(allowed_ids)} unique items)")

    return df

def perform_5_core_filtering(df, min_u_num=8, min_i_num=1):
    """Perform 5-core filtering on the dataset"""
    print(f"Original dataset shape: {df.shape}")
    
    # Clean data
    df.dropna(subset=['userID', 'itemID', 'timestamp'], inplace=True)
    df.drop_duplicates(subset=['userID', 'itemID', 'timestamp'], inplace=True)
    print(f"After cleaning: {df.shape}")
    
    # 5-core filtering
    k_core = 5
    learner_id, course_id = 'userID', 'itemID'
    
    def get_illegal_ids_by_inter_num(df, field, max_num=None, min_num=None):
        if field is None:
            return set()
        if max_num is None and min_num is None:
            return set()

        max_num = max_num or np.inf
        min_num = min_num or -1

        ids = df[field].values
        inter_num = Counter(ids)
        ids = {id_ for id_ in inter_num if inter_num[id_] < min_num or inter_num[id_] > max_num}
        print(f'{len(ids)} illegal_ids_by_inter_num, field={field}')

        return ids

    def filter_by_k_core(df):
        while True:
            ban_users = get_illegal_ids_by_inter_num(df, field=learner_id, max_num=None, min_num=min_u_num)
            ban_items = get_illegal_ids_by_inter_num(df, field=course_id, max_num=None, min_num=min_i_num)
            if len(ban_users) == 0 and len(ban_items) == 0:
                return

            dropped_inter = pd.Series(False, index=df.index)
            if learner_id:
                dropped_inter |= df[learner_id].isin(ban_users)
            if course_id:
                dropped_inter |= df[course_id].isin(ban_items)
            print(f'{len(dropped_inter)} dropped interactions')
            df.drop(df.index[dropped_inter], inplace=True)

    filter_by_k_core(df)
    print(f"After 5-core filtering: {df.shape}")
    return df

def reindex_users_and_items(df):
    """Reindex users and items with sequential IDs"""
    # Create user and item mappings
    unique_users = sorted(df['userID'].unique())
    unique_items = sorted(df['itemID'].unique())
    
    user_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
    item_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_items)}
    
    # Apply mappings
    df['userID'] = df['userID'].map(user_id_map)
    df['itemID'] = df['itemID'].map(item_id_map)
    
    print(f"Reindexed {len(unique_users)} users and {len(unique_items)} items")
    return df, user_id_map, item_id_map

def split_train_valid_test(df, output_dir, category_name):
    """Split data into train/validation/test sets"""
    # Sort by user and timestamp
    df = df.sort_values(['userID', 'timestamp']).reset_index(drop=True)
    
    # Group by user
    uid_field, iid_field = 'userID', 'itemID'
    uid_freq = df.groupby(uid_field)[iid_field]
    u_i_dict = {}
    for u, u_ls in uid_freq:
        u_i_dict[u] = list(u_ls)
    
    # Create labels for train/valid/test split
    new_label = []
    u_ids_sorted = sorted(u_i_dict.keys())

    for u in u_ids_sorted:
        items = u_i_dict[u]
        n_items = len(items)
        if n_items < 10:
            tmp_ls = [0] * (n_items - 2) + [1] + [2]
        else:
            val_test_len = int(n_items * 0.2)
            train_len = n_items - val_test_len
            val_len = val_test_len // 2
            test_len = val_test_len - val_len
            tmp_ls = [0] * train_len + [1] * val_len + [2] * test_len
        new_label.extend(tmp_ls)

    df['x_label'] = new_label
    
    # Split into separate files
    train_df = df[df['x_label'] == 0].drop('x_label', axis=1)
    valid_df = df[df['x_label'] == 1].drop('x_label', axis=1)
    test_df = df[df['x_label'] == 2].drop('x_label', axis=1)
    
    # Save files
    train_file = os.path.join(output_dir, f'{category_name.lower()}-train.inter')
    valid_file = os.path.join(output_dir, f'{category_name.lower()}-valid.inter')
    test_file = os.path.join(output_dir, f'{category_name.lower()}-test.inter')
    
    train_df.to_csv(train_file, sep='	', index=False)
    valid_df.to_csv(valid_file, sep='	', index=False)
    test_df.to_csv(test_file, sep='	', index=False)
    
    print(f"Split complete:")
    print(f"  Train: {len(train_df)} interactions")
    print(f"  Valid: {len(valid_df)} interactions")
    print(f"  Test: {len(test_df)} interactions")
    
    return df

def process_category(category_name, output_dir):
    """Process a single category"""
    print(f"\n{'='*50}")
    print(f"Processing category: {category_name}")
    print(f"{'='*50}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Check if the category has already been processed
    final_output_file = os.path.join(output_dir, 'i_id_mapping.csv')
    if os.path.exists(final_output_file):
        print(f"Category '{category_name}' has already been processed. Skipping.")
        return
    
    # Load raw DataFrame (with caching to avoid repeated downloads/conversions)
    try:
        df = load_or_cache_dataframe(category_name, output_dir)
    except RuntimeError as e:
        print(e)
        return

    # Filter by item whitelist specific to the category
    df = filter_df_by_item_whitelist(df, category_name)
    
    # Perform 5-core filtering
    df = perform_5_core_filtering(df)
    
    # Reindex users and items
    df, user_id_map, item_id_map = reindex_users_and_items(df)
    
    # Add x_label and split data into train/valid/test files
    df = split_train_valid_test(df, output_dir, category_name)

    # Save reindexed data
    indexed_file = os.path.join(output_dir, f'{category_name.lower()}.inter')
    df.to_csv(indexed_file, sep='	', index=False)
    
    # Save mappings as CSV files
    user_mapping_df = pd.DataFrame(list(user_id_map.items()), columns=['original_id', 'new_id'])
    item_mapping_df = pd.DataFrame(list(item_id_map.items()), columns=['original_id', 'new_id'])
    
    user_mapping_file = os.path.join(output_dir, f'u_id_mapping.csv')
    item_mapping_file = os.path.join(output_dir, f'i_id_mapping.csv')
    
    user_mapping_df.to_csv(user_mapping_file, index=False)
    item_mapping_df.to_csv(item_mapping_file, index=False)
    
    print(f"Processing complete for {category_name}")
    print(f"Output files saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Process Amazon review datasets')
    parser.add_argument('--category', type=str, default='All_Beauty', 
                       help='Category to process (default: All_Beauty)')
    parser.add_argument('--output_dir', type=str, default='/data/junch/MMRec/data',
                       help='Output directory for processed data')
    parser.add_argument('--all_categories', action='store_true',
                       help='Process all available categories')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.all_categories:
        # Process all categories
        categories = list(get_amazon_category_mapping().keys())
        for category in categories:
            category_output_dir = os.path.join(args.output_dir, category.lower())
            process_category(category, category_output_dir)
    else:
        # Process single category
        category_output_dir = os.path.join(args.output_dir, args.category.lower())
        process_category(args.category, category_output_dir)

if __name__ == "__main__":
    main()
