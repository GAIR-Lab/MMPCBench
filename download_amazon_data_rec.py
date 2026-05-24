import json
import os
import time
import requests
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import logging
from collections import defaultdict, Counter
import random
import pickle

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('amazon_data_rec.log'),
        logging.StreamHandler()
    ]
)

class AmazonDataRecCollector:
    """Optimized Amazon data collector for multimodal recommendation."""
    
    def __init__(self, output_dir: str = "amazon_data_rec"):
        """
        Initialize the data collector.
        
        Args:
            output_dir: Output directory for collected data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Target categories
        self.categories = [
            "All_Beauty", "Toys_and_Games", "Video_Games", "Home_and_Kitchen",
            "Electronics", "Industrial_and_Scientific", "Office_Products", 
            "Musical_Instruments", "Arts_Crafts_and_Sewing"
        ]
        
        # Target users per category
        self.target_users_per_category = 1000
        
        # Target items per category (minimize this)
        self.target_items_per_category = 300  # Adjust item count to support 1000 users
        
        # Minimum interactions per user
        self.min_interactions_per_user = 5
        self.max_interactions_per_user = 20
        
        logging.info(f"Initialized collector for {len(self.categories)} categories")
    
    def download_image(self, url, filepath, max_retries=3):
        """Download image from URL to filepath with retry logic"""
        for attempt in range(max_retries):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, timeout=30, stream=True, headers=headers)
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logging.error(f"Failed to download {url} after {max_retries} attempts")
                    return False
    
    def collect_category_data(self, category_name: str):
        """Collect optimized data for a single category - USER FIRST approach."""
        logging.info(f"Collecting data for category: {category_name}")
        
        # Create category directory
        category_dir = self.output_dir / category_name
        category_dir.mkdir(exist_ok=True)
        (category_dir / "images").mkdir(exist_ok=True)
        
        try:
            # Load datasets
            logging.info(f"Loading {category_name} datasets...")
            dataset_meta = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{category_name}", split="full", trust_remote_code=True)
            dataset_review = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{category_name}", split="full", trust_remote_code=True)
            
            # Step 1: Collect item metadata for potential modal information
            logging.info("Collecting item metadata...")
            item_metadata_raw = {}
            items_with_modal_info = set()
            
            for item in tqdm(dataset_meta, desc=f"Processing {category_name} metadata"):
                parent_asin = item.get("parent_asin", "")
                title = item.get("title", "")
                
                # Check if item has title
                if not title or len(title.strip()) < 5:
                    continue
                
                # Check if item has image URL
                images = item.get("images", {})
                large_images = images.get("large", []) if images else []
                image_url = None
                for img in large_images:
                    if img and img.startswith('http'):
                        image_url = img
                        break
                
                if not image_url:
                    continue
                
                # Item has potential modal information (title + image URL)
                items_with_modal_info.add(parent_asin)
                
                categories = item.get("categories", [])
                store = item.get("store", "")
                details = item.get("details", "")
                price = item.get("price", "")
                
                item_metadata_raw[parent_asin] = {
                    'title': title,
                    'categories': categories,
                    'brand': str(store) if store else "",
                    'details': str(details) if details else "",
                    'price': str(price) if price else "",
                    'image_url': image_url
                }
            
            logging.info(f"Found {len(items_with_modal_info)} items with potential modal information")
            
            # Step 2: Build user interaction profiles (FOCUS ON USERS FIRST)
            logging.info("Building user interaction profiles...")
            user_item_interactions = defaultdict(list)
            
            for review in tqdm(dataset_review, desc=f"Processing {category_name} user interactions"):
                parent_asin = review.get("parent_asin", "")
                user_id = review.get("user_id", "")
                timestamp = review.get("timestamp", 0)
                rating = review.get("rating", 0)
                
                # Only consider items with modal info and positive reviews
                if parent_asin in items_with_modal_info and rating >= 4:
                    user_item_interactions[user_id].append((parent_asin, timestamp))
            
            # Step 3: Filter users with sufficient interactions
            logging.info("Filtering users with sufficient interactions...")
            valid_users = {}
            for user_id, interactions in user_item_interactions.items():
                if len(interactions) >= self.min_interactions_per_user:
                    # Sort by timestamp
                    sorted_interactions = sorted(interactions, key=lambda x: x[1])
                    # If sequence is longer than max, truncate to first 20 items
                    if len(sorted_interactions) > self.max_interactions_per_user:
                        sorted_interactions = sorted_interactions[:self.max_interactions_per_user]
                    valid_users[user_id] = [item_id for item_id, _ in sorted_interactions]
            
            logging.info(f"Found {len(valid_users)} valid users for {category_name}")
            
            if len(valid_users) < self.target_users_per_category:
                logging.warning(f"Not enough valid users found: {len(valid_users)} < {self.target_users_per_category}")
                logging.info(f"Continuing with {len(valid_users)} users instead of target {self.target_users_per_category}")
                # Adjust target to actual available users
                actual_target = min(len(valid_users), self.target_users_per_category)
            
            # Step 4: Select users to minimize item overlap
            actual_target = min(len(valid_users), self.target_users_per_category)
            logging.info(f"Selecting exactly {actual_target} users to minimize item overlap...")
            selected_users = self._select_users_with_minimal_overlap(valid_users, actual_target)
            
            logging.info(f"Selected {len(selected_users)} users for {category_name}")
            
            # Step 5: Find all items used by selected users
            logging.info("Finding all items used by selected users...")
            required_items = set()
            for user_interactions in selected_users.values():
                required_items.update(user_interactions)
            
            logging.info(f"Selected users require {len(required_items)} unique items")
            
            # Step 6: Download images for all required items
            logging.info("Downloading images for all required items...")
            final_item_metadata = {}
            downloaded_count = 0
            failed_items = set()
            
            for item_id in tqdm(required_items, desc=f"Downloading images for {category_name}"):
                if item_id not in item_metadata_raw:
                    failed_items.add(item_id)
                    continue
                
                item_info = item_metadata_raw[item_id]
                image_url = item_info['image_url']
                
                # Download image
                image_filename = f"{item_id}.jpg"
                image_filepath = category_dir / "images" / image_filename
                
                if self.download_image(image_url, image_filepath):
                    downloaded_count += 1
                    
                    # Add to final metadata
                    final_item_metadata[item_id] = {
                        'title': item_info['title'],
                        'categories': item_info['categories'],
                        'brand': item_info['brand'],
                        'details': item_info['details'],
                        'price': item_info['price'],
                        'image_path': f"images/{image_filename}"
                    }
                else:
                    failed_items.add(item_id)
            
            logging.info(f"Successfully downloaded {downloaded_count}/{len(required_items)} images")
            logging.info(f"Failed to download {len(failed_items)} images")
            
            # Step 7: Filter users by successfully downloaded items
            logging.info("Filtering users by successfully downloaded items...")
            final_valid_users = {}
            for user_id, interactions in selected_users.items():
                # Filter interactions to only include successfully downloaded items
                final_interactions = [item_id for item_id in interactions if item_id in final_item_metadata]
                # Apply same truncation rule: keep up to max_interactions_per_user
                if len(final_interactions) > self.max_interactions_per_user:
                    final_interactions = final_interactions[:self.max_interactions_per_user]
                if len(final_interactions) >= self.min_interactions_per_user:
                    final_valid_users[user_id] = final_interactions
            
            logging.info(f"After filtering by downloaded items: {len(final_valid_users)} users")
            
            # Step 8: If we lost too many users, find replacements
            if len(final_valid_users) < actual_target * 0.9:  # If we lost more than 10%
                logging.info(f"Need to find replacement users...")
                needed_users = actual_target - len(final_valid_users)
                additional_users = self._find_replacement_users(valid_users, final_item_metadata, 
                                                              set(final_valid_users.keys()), needed_users)
                final_valid_users.update(additional_users)
                logging.info(f"After finding replacements: {len(final_valid_users)} users")
            
            if len(final_valid_users) == 0:
                logging.error(f"No valid users found for {category_name}")
                # Continue to next category instead of returning
                return
            
            # Step 10: Save data in MSRBench format
            self._save_msrbench_format(category_dir, final_item_metadata, final_valid_users, category_name)
            
            # Step 9: Save summary
            summary = {
                'category': category_name,
                'num_items': len(final_item_metadata),
                'num_users': len(final_valid_users),
                'total_interactions': sum(len(seq) for seq in final_valid_users.values()),
                'avg_sequence_length': sum(len(seq) for seq in final_valid_users.values()) / len(final_valid_users) if final_valid_users else 0,
                'items_with_modal_info': len(items_with_modal_info),
                'required_items': len(required_items),
                'downloaded_images': downloaded_count,
                'failed_images': len(failed_items)
            }
            
            with open(category_dir / "summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
            logging.info(f"Category {category_name} completed: {summary}")
            
        except Exception as e:
            logging.error(f"Error processing category {category_name}: {e}")
    
    def _select_users_with_minimal_overlap(self, valid_users: dict, target_count: int):
        """Select users with minimal item overlap to ensure we get exactly target_count users."""
        # Sort users by number of interactions (prefer users with fewer interactions)
        sorted_users = sorted(valid_users.items(), key=lambda x: len(x[1]))
        
        selected_users = {}
        covered_items = set()
        
        # First pass: try to get users with minimal new items
        for user_id, interactions in sorted_users:
            if len(selected_users) >= target_count:
                break
            
            new_items = set(interactions) - covered_items
            # Allow up to 20 new items per user initially
            if len(new_items) <= 20:
                selected_users[user_id] = interactions
                covered_items.update(interactions)
        
        # Second pass: if we don't have enough users, be more permissive
        if len(selected_users) < target_count * 0.8:
            for user_id, interactions in sorted_users:
                if user_id not in selected_users and len(selected_users) < target_count:
                    new_items = set(interactions) - covered_items
                    # Allow up to 40 new items per user
                    if len(new_items) <= 40:
                        selected_users[user_id] = interactions
                        covered_items.update(interactions)
        
        # Third pass: if still not enough users, accept any user
        if len(selected_users) < target_count * 0.9:
            for user_id, interactions in sorted_users:
                if user_id not in selected_users and len(selected_users) < target_count:
                    selected_users[user_id] = interactions
                    covered_items.update(interactions)
        
        # Fourth pass: if still not enough, take any remaining users
        if len(selected_users) < target_count:
            for user_id, interactions in sorted_users:
                if user_id not in selected_users and len(selected_users) < target_count:
                    selected_users[user_id] = interactions
                    covered_items.update(interactions)
        
        logging.info(f"Selected {len(selected_users)} users covering {len(covered_items)} items")
        return selected_users
    
    def _find_replacement_users(self, valid_users: dict, final_items: set, existing_users: set, needed_count: int):
        """Find replacement users to reach the target count."""
        replacement_users = {}
        
        # Sort users by how many final items they interact with
        user_scores = {}
        for user_id, interactions in valid_users.items():
            if user_id in existing_users:  # Skip already selected users
                continue
                
            final_interactions = [item_id for item_id in interactions if item_id in final_items]
            # Apply same truncation rule: keep up to max_interactions_per_user
            if len(final_interactions) > self.max_interactions_per_user:
                final_interactions = final_interactions[:self.max_interactions_per_user]
            if len(final_interactions) >= self.min_interactions_per_user:
                user_scores[user_id] = len(final_interactions)
        
        # Sort by score (prefer users with more final item interactions)
        sorted_users = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)
        
        for user_id, score in sorted_users:
            if len(replacement_users) >= needed_count:
                break
            
            interactions = valid_users[user_id]
            final_interactions = [item_id for item_id in interactions if item_id in final_items]
            # Apply same truncation rule: keep up to max_interactions_per_user
            if len(final_interactions) > self.max_interactions_per_user:
                final_interactions = final_interactions[:self.max_interactions_per_user]
            replacement_users[user_id] = final_interactions
        
        logging.info(f"Found {len(replacement_users)} replacement users")
        return replacement_users
    
    def _save_msrbench_format(self, category_dir: Path, item_metadata: dict, user_sequences: dict, category_name: str):
        """Save data in MSRBench format."""
        
        # 1. Build mappings
        item2id = {}
        id2item = {}
        user2id = {}
        id2user = {}
        
        # Item mappings
        for item_id in item_metadata.keys():
            if item_id not in item2id:
                item_idx = len(item2id)
                item2id[item_id] = str(item_idx)
                id2item[str(item_idx)] = item_id
        
        # User mappings
        for user_id in user_sequences.keys():
            if user_id not in user2id:
                user_idx = len(user2id)
                user2id[user_id] = str(user_idx)
                id2user[str(user_idx)] = user_id
        
        mappings = {
            'item2id': item2id,
            'id2item': id2item,
            'user2id': user2id,
            'id2user': id2user
        }
        
        # 2. Build item2side.json
        item2side = {}
        for item_id, metadata in item_metadata.items():
            item2side[item_id] = {
                'title': metadata['title'],
                'categories': metadata['categories'],
                'brand': metadata['brand'],
                'details': metadata['details'],
                'price': metadata['price'],
                'image': {
                    'image_path': metadata['image_path']
                }
            }
        
        # 3. Build purchase_history.pkl (MSRBench format)
        purchase_history = []
        for user_id, sequence in user_sequences.items():
            # Convert item IDs to internal IDs
            internal_sequence = [item2id[item_id] for item_id in sequence if item_id in item2id]
            if internal_sequence:
                purchase_history.append(internal_sequence)
        
        # 4. Build negative_samples.txt
        user2neg = {}
        all_items = set(item2id.keys())
        
        for user_id, sequence in user_sequences.items():
            user_items = set(sequence)
            negative_items = list(all_items - user_items)
            # Sample up to 50 negative items per user (reduced from 100)
            if len(negative_items) > 50:
                negative_items = random.sample(negative_items, 50)
            user2neg[user_id] = negative_items
        
        # Save files
        with open(category_dir / "datamaps.json", 'w', encoding='utf-8') as f:
            json.dump(mappings, f, indent=2, ensure_ascii=False)
        
        with open(category_dir / "item2side.json", 'w', encoding='utf-8') as f:
            json.dump(item2side, f, indent=2, ensure_ascii=False)
        
        with open(category_dir / "purchase_history.pkl", 'wb') as f:
            pickle.dump(purchase_history, f)
        
        with open(category_dir / "negative_samples.txt", 'w', encoding='utf-8') as f:
            for user_id, negative_items in user2neg.items():
                line = f"{user_id} {' '.join(negative_items)}\n"
                f.write(line)
        
        logging.info(f"MSRBench format files saved to {category_dir}")
    
    def collect_all_categories(self):
        """Collect data for all categories."""
        logging.info("Starting data collection for all categories...")
        
        for i, category in enumerate(self.categories, 1):
            logging.info(f"Processing category {i}/{len(self.categories)}: {category}")
            try:
                self.collect_category_data(category)
            except Exception as e:
                logging.error(f"Failed to process category {category}: {e}")
                continue
        
        logging.info("All categories processed!")
        
        # Generate overall summary
        self._generate_overall_summary()
    
    def _generate_overall_summary(self):
        """Generate overall summary of all categories."""
        overall_summary = {
            'total_categories': len(self.categories),
            'categories': {}
        }
        
        total_items = 0
        total_users = 0
        total_interactions = 0
        
        for category in self.categories:
            category_dir = self.output_dir / category
            summary_file = category_dir / "summary.json"
            
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    overall_summary['categories'][category] = summary
                    
                    total_items += summary['num_items']
                    total_users += summary['num_users']
                    total_interactions += summary['total_interactions']
        
        overall_summary['total_items'] = total_items
        overall_summary['total_users'] = total_users
        overall_summary['total_interactions'] = total_interactions
        overall_summary['avg_items_per_category'] = total_items / len(self.categories)
        overall_summary['avg_users_per_category'] = total_users / len(self.categories)
        
        with open(self.output_dir / "overall_summary.json", 'w') as f:
            json.dump(overall_summary, f, indent=2)
        
        logging.info(f"Overall summary: {total_items} items, {total_users} users, {total_interactions} interactions")

def main():
    """Main function."""
    output_dir = "amazon_data_rec"
    
    collector = AmazonDataRecCollector(output_dir)
    collector.collect_all_categories()

if __name__ == "__main__":
    main() 