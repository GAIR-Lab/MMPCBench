#!/usr/bin/env python3
"""
MMRec Experiment Runner Script
Run VBPR, MMGCN, FREEDOM three models on 9 Amazon categories
"""

import os
import subprocess
import time
from datetime import datetime

# Configuration
GPU_ID = 3  # Modify to your GPU ID
MODELS = ['FREEDOM'，'VBPR','BM3']
DATASETS = [
    'all_beauty',
    'toys_and_games', 
    'video_games',
    'home_and_kitchen',
    'electronics',
    'industrial_and_scientific',
    'office_products',
    'musical_instruments',
    'arts_crafts_and_sewing'
]

def run_experiment(model, dataset):
    """Run a single experiment"""
    cmd = f"python main.py -m {model} -d {dataset}"
    
    print(f"\n{'='*60}")
    print(f"Running: {model} on {dataset}")
    print(f"Command: {cmd}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Set GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
    
    try:
        # Run experiment with 2-hour timeout
        result = subprocess.run(
            cmd.split(),
            env=env,
            timeout=7200,
            capture_output=False  # Real-time output display
        )
        
        if result.returncode == 0:
            print(f"✅ Success: {model} on {dataset}")
        else:
            print(f"❌ Failed: {model} on {dataset}")
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout: {model} on {dataset} (2 hours)")
    except Exception as e:
        print(f"💥 Error: {model} on {dataset} - {e}")

def main():
    print("MMRec Batch Experiments")
    print(f"GPU: {GPU_ID}")
    print(f"Models: {MODELS}")
    print(f"Datasets: {DATASETS}")
    print(f"Total experiments: {len(MODELS) * len(DATASETS)}")
    
    total = len(MODELS) * len(DATASETS)
    current = 0
    
    start_time = time.time()
    
    # Run all experiments
    for model in MODELS:
        for dataset in DATASETS:
            current += 1
            print(f"\n[{current}/{total}] Starting experiment...")
            run_experiment(model, dataset)
            
            # Experiment interval
            time.sleep(3)
    
    end_time = time.time()
    duration = (end_time - start_time) / 3600  # Hours
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Total time: {duration:.2f} hours")
    print(f"{'='*60}")

if __name__ == '__main__':
    main() 