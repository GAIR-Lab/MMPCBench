import json
import csv
import os
import glob
import re

def extract_model_name_from_path(file_path):
    """Extract model name from file path using regex patterns"""
    # Pattern to match qwen_vl_Xb or gemma_vl_Xb where X is any number
    qwen_pattern = r'qwen_vl_\d+b'
    gemma_pattern = r'gemma_vl_\d+b'
    
    # Pattern to match qwen_vl_unknown or gemma_vl_unknown
    qwen_unknown_pattern = r'qwen_vl_unknown'
    gemma_unknown_pattern = r'gemma_vl_unknown'
    
    # Search for patterns in the file path
    qwen_match = re.search(qwen_pattern, file_path, re.IGNORECASE)
    gemma_match = re.search(gemma_pattern, file_path, re.IGNORECASE)
    qwen_unknown_match = re.search(qwen_unknown_pattern, file_path, re.IGNORECASE)
    gemma_unknown_match = re.search(gemma_unknown_pattern, file_path, re.IGNORECASE)
    
    if qwen_match:
        return qwen_match.group().lower()
    elif gemma_match:
        return gemma_match.group().lower()
    elif qwen_unknown_match:
        return qwen_unknown_match.group().lower()
    elif gemma_unknown_match:
        return gemma_unknown_match.group().lower()
    else:
        return 'unknown'

def process_jsonl_file(input_file, output_dir):
    """Process a single jsonl file and generate CSV and TSV outputs"""
    rows = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            
            # Extract data based on file type
            item_id = obj.get('image', '')
            
            if 'titles' in input_file:
                # For title files, extract only the title
                title = obj.get('title', '')
                rows.append([item_id, title])
            elif 'prompts' in input_file:
                # For prompt files, extract prompt and negative_prompt
                prompt = obj.get('prompt', '')
                negative_prompt = obj.get('negative_prompt', '')
                rows.append([item_id, prompt, negative_prompt])
    
    # Determine model and type from input file path using regex
    model_name = extract_model_name_from_path(input_file)
    
    if 'titles' in input_file:
        data_type = 'titles'
    elif 'prompts' in input_file:
        data_type = 'prompts'
    else:
        data_type = 'unknown'
    
    # Create model-specific output directory
    model_output_dir = os.path.join(output_dir, model_name, data_type)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Generate output filenames (simplified without model suffix since it's in folder)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    # Remove the model suffix from filename since it's now in the folder structure
    if model_name in base_name:
        base_name = base_name.replace(f'_{model_name}', '')
    if data_type in base_name:
        base_name = base_name.replace(f'_{data_type}', '')
    
    csv_output = os.path.join(model_output_dir, f"{base_name}.csv")
    tsv_output = os.path.join(model_output_dir, f"{base_name}.tsv")
    
    # Write CSV
    with open(csv_output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if 'titles' in input_file:
            writer.writerow(['item_id', 'title'])
        else:
            writer.writerow(['item_id', 'prompt', 'negative_prompt'])
        writer.writerows(rows)
    
    # Write TSV
    with open(tsv_output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        if 'titles' in input_file:
            writer.writerow(['item_id', 'title'])
        else:
            writer.writerow(['item_id', 'prompt', 'negative_prompt'])
        writer.writerows(rows)
    
    # Enhanced output formatting
    print(f"✓ Processed: {os.path.basename(input_file)}")
    print(f"  └─ Model: {model_name}")
    print(f"  └─ Type: {data_type}")
    print(f"  └─ Rows: {len(rows)}")
    print(f"  └─ Output: {os.path.basename(csv_output)} | {os.path.basename(tsv_output)}")
    print(f"  └─ Location: {model_output_dir}")
    print()

def find_all_model_directories():
    """Find all directories that match the pattern outputs_[model]_[type]"""
    # Updated pattern to include both numbered versions (e.g., qwen_vl_32b) and unknown versions (e.g., qwen_vl_unknown)
    pattern = r'outputs_(qwen_vl_(\d+b|unknown)|gemma_vl_(\d+b|unknown))_(titles|prompts)'
    directories = []
    
    for item in os.listdir('.'):
        if os.path.isdir(item) and re.match(pattern, item, re.IGNORECASE):
            directories.append(item)
    
    return sorted(directories)

def main():
    print("🚀 Starting JSONL to CSV/TSV conversion with organized folder structure...")
    print("=" * 70)
    
    # Automatically find all model directories
    directories = find_all_model_directories()
    
    if not directories:
        print("❌ No model directories found!")
        print("Expected pattern: outputs_[model]_[type]")
        print("Examples: outputs_qwen_vl_32b_titles, outputs_gemma_vl_27b_prompts, outputs_qwen_vl_unknown_titles")
        return
    
    print(f"📁 Found {len(directories)} directories to process:")
    for dir in directories:
        print(f"  - {dir}")
    print()
    
    # Create main output directory
    output_dir = 'organized_outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    total_files = 0
    processed_models = set()
    
    # Process each directory
    for directory in directories:
        if os.path.exists(directory):
            print(f"📁 Processing directory: {directory}")
            jsonl_files = glob.glob(os.path.join(directory, "*.jsonl"))
            
            if not jsonl_files:
                print(f"  ⚠️  No JSONL files found in {directory}")
                continue
                
            for jsonl_file in jsonl_files:
                process_jsonl_file(jsonl_file, output_dir)
                total_files += 1
                # Track processed models for summary
                model_name = extract_model_name_from_path(jsonl_file)
                processed_models.add(model_name)
        else:
            print(f"❌ Directory not found: {directory}")
    
    print("=" * 70)
    print(f"✅ Processing complete!")
    print(f"📊 Total files processed: {total_files}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Generated: {total_files * 2} files (CSV + TSV for each JSONL)")
    print()
    print("📂 Folder structure:")
    print(f"  {output_dir}/")
    
    # Show actual processed models in the structure
    for model in sorted(processed_models):
        print(f"  ├── {model}/")
        print(f"  │   ├── titles/")
        print(f"  │   └── prompts/")
    
    print()
    print("🎯 Supported model variants:")
    print("  - qwen_vl_4b, qwen_vl_7b, qwen_vl_32b, qwen_vl_72b, qwen_vl_unknown, etc.")
    print("  - gemma_vl_4b, gemma_vl_12b, gemma_vl_27b, gemma_vl_unknown, etc.")

if __name__ == "__main__":
    main()
