import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch
import lpips
import json
from datetime import datetime
import argparse
from tqdm import tqdm
from PIL import Image
import warnings
import clip
from pytorch_fid import fid_score
import tempfile
import shutil

# Suppress libpng warnings
warnings.filterwarnings('ignore', category=UserWarning, module='PIL')

# Initialize CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Define all models and categories
MODELS = [
    'gemma_vl_4b',
    'gemma_vl_12b', 
    'gemma_vl_27b',
    'qwen_vl_7b',
    'qwen_vl_32b',
    'qwen_vl_72b'
]

CATEGORIES = [
    'All_Beauty',
    'Toys_and_Games', 
    'Video_Games',
    'Home_and_Kitchen',
    'Electronics',
    'Industrial_and_Scientific',
    'Office_Products',
    'Musical_Instruments',
    'Arts_Crafts_and_Sewing'
]

# Initialize LPIPS model
lpips_model = lpips.LPIPS(net='alex')

def load_image_with_fallback(file_path):
    """Load image using OpenCV first, then PIL as fallback."""
    # Try OpenCV first
    img = cv2.imread(file_path)
    if img is not None:
        return img
    
    # If OpenCV fails, try PIL
    try:
        with Image.open(file_path) as pil_img:
            # Convert to RGB if needed
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            # Convert to numpy array
            img_array = np.array(pil_img)
            
            # Convert RGB to BGR for OpenCV compatibility
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            return img_bgr
            
    except Exception as e:
        return None

def diagnose_image_file(file_path):
    """Diagnose issues with image files and return detailed error information."""
    if not os.path.exists(file_path):
        return f"File does not exist: {file_path}"
    
    try:
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return f"File is empty (0 bytes): {file_path}"
        
        # Try to read the file header to check if it's a valid image
        with open(file_path, 'rb') as f:
            header = f.read(10)
            
        # Check for common image format headers
        if header.startswith(b'\xff\xd8\xff'):  # JPEG
            return None  # Valid JPEG
        elif header.startswith(b'\x89PNG\r\n\x1a\n'):  # PNG
            return None  # Valid PNG
        elif header.startswith(b'GIF87a') or header.startswith(b'GIF89a'):  # GIF
            return None  # Valid GIF
        elif header.startswith(b'BM'):  # BMP
            return None  # Valid BMP
        else:
            return f"Unknown or corrupted image format: {file_path} (header: {header[:8].hex()})"
            
    except OSError as e:
        return f"Cannot access file: {file_path} - {e}"
    except Exception as e:
        return f"Unexpected error reading file: {file_path} - {e}"

def calculate_metrics(img1, img2):
    """Calculate PSNR, SSIM, MSE, and LPIPS between two images."""
    # Ensure images are float32 and in [0, 1]
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0
    
    # PSNR
    psnr_val = psnr(img1, img2, data_range=1.0)
    
    # SSIM
    min_dim = min(img1.shape[0], img1.shape[1])
    win_size = 7 if min_dim >= 7 else (min_dim if min_dim % 2 == 1 else min_dim - 1)
    ssim_val = ssim(img1, img2, data_range=1.0, channel_axis=-1, win_size=win_size)
    
    # MSE
    mse_val = np.mean((img1 - img2) ** 2)
    
    # LPIPS
    if img1.shape[0] >= 64 and img1.shape[1] >= 64:
        img1_t = torch.from_numpy(img1).permute(2,0,1).unsqueeze(0) * 2 - 1
        img2_t = torch.from_numpy(img2).permute(2,0,1).unsqueeze(0) * 2 - 1
        with torch.no_grad():
            lpips_val = lpips_model(img1_t, img2_t).item()
    else:
        lpips_val = None
        
    return psnr_val, ssim_val, mse_val, lpips_val

def calculate_clip_score(img1, img2):
    """Calculate CLIP Score between two images."""
    try:
        # Ensure images are valid
        if img1 is None or img2 is None:
            return None
            
        # Convert BGR to RGB for CLIP
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        img1_pil = Image.fromarray(img1_rgb)
        img2_pil = Image.fromarray(img2_rgb)
        
        # Preprocess images for CLIP
        img1_tensor = clip_preprocess(img1_pil).unsqueeze(0).to(device)
        img2_tensor = clip_preprocess(img2_pil).unsqueeze(0).to(device)
        
        # Get CLIP features
        with torch.no_grad():
            features1 = clip_model.encode_image(img1_tensor)
            features2 = clip_model.encode_image(img2_tensor)
            
            # Normalize features
            features1 = features1 / features1.norm(dim=-1, keepdim=True)
            features2 = features2 / features2.norm(dim=-1, keepdim=True)
            
            # Calculate cosine similarity (CLIP Score)
            clip_score = torch.cosine_similarity(features1, features2).item()
            
        return clip_score
        
    except Exception as e:
        print(f"Error calculating CLIP Score: {e}")
        return None

def calculate_fid_score(real_images, generated_images, temp_dir=None):
    """Calculate FID Score between real and generated images."""
    try:
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
        
        real_dir = os.path.join(temp_dir, "real")
        gen_dir = os.path.join(temp_dir, "generated")
        
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(gen_dir, exist_ok=True)
        
        # Save images to temporary directories with consistent sizing
        saved_count = 0
        for i, (real_img, gen_img) in enumerate(zip(real_images, generated_images)):
            if real_img is not None and gen_img is not None:
                try:
                    # Resize images to a consistent size (299x299 for Inception)
                    target_size = (299, 299)
                    
                    # Resize real image
                    real_resized = cv2.resize(real_img, target_size)
                    real_path = os.path.join(real_dir, f"real_{i:06d}.png")
                    success_real = cv2.imwrite(real_path, real_resized)
                    
                    # Resize generated image
                    gen_resized = cv2.resize(gen_img, target_size)
                    gen_path = os.path.join(gen_dir, f"gen_{i:06d}.png")
                    success_gen = cv2.imwrite(gen_path, gen_resized)
                    
                    if success_real and success_gen:
                        saved_count += 1
                        
                except Exception as e:
                    print(f"Warning: Failed to save image pair {i}: {e}")
                    continue
        
        if saved_count < 10:
            print(f"Warning: Only {saved_count} image pairs saved, FID calculation may be unreliable")
            return None
        
        # Calculate FID score with smaller batch size and error handling
        try:
            fid_value = fid_score.calculate_fid_given_paths([real_dir, gen_dir], 
                                                           batch_size=16,  # Reduced batch size
                                                           device=device,
                                                           dims=2048,
                                                           num_workers=0)  # Disable multiprocessing
            return fid_value
            
        except Exception as e:
            print(f"Error in FID calculation: {e}")
            # Try with even smaller batch size
            try:
                fid_value = fid_score.calculate_fid_given_paths([real_dir, gen_dir], 
                                                               batch_size=8, 
                                                               device=device,
                                                               dims=2048,
                                                               num_workers=0)
                return fid_value
            except Exception as e2:
                print(f"FID calculation failed even with smaller batch size: {e2}")
                return None
        
    except Exception as e:
        print(f"Error calculating FID Score: {e}")
        return None
    finally:
        # Clean up temporary directory if we created it
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Failed to clean up temp directory: {e}")

def evaluate_model_category(model, category, gen_folder, real_folder, verbose=True, skip_fid=False):
    """Evaluate a specific model on a specific category."""
    gen_path = os.path.join(gen_folder, model, category)
    real_path = os.path.join(real_folder, category, 'images')
    
    if not os.path.exists(gen_path):
        if verbose:
            print(f"Warning: Generated images path does not exist: {gen_path}")
        return None
    
    if not os.path.exists(real_path):
        if verbose:
            print(f"Warning: Real images path does not exist: {real_path}")
        return None
    
    # Get all image files
    gen_files = sorted([f for f in os.listdir(gen_path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    real_files = sorted([f for f in os.listdir(real_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Match by filename
    files = list(set(gen_files) & set(real_files))
    if not files:
        if verbose:
            print(f"No matching files found for {model}/{category}")
        return None
    
    psnr_list, ssim_list, mse_list, lpips_list, clip_list = [], [], [], [], []
    valid_files = 0
    real_images_for_fid = []
    gen_images_for_fid = []
    
    # Add progress bar for file processing
    pbar = tqdm(files, desc=f"Processing {model}/{category}", unit="file") if verbose else files
    
    for fname in pbar:
        gen_img_path = os.path.join(gen_path, fname)
        real_img_path = os.path.join(real_path, fname)
        
        # Diagnose image files
        gen_diagnosis = diagnose_image_file(gen_img_path)
        real_diagnosis = diagnose_image_file(real_img_path)
        
        if gen_diagnosis:
            if verbose:
                print(f'\nSkipping {fname} (generated image issue: {gen_diagnosis})')
            continue
            
        if real_diagnosis:
            if verbose:
                print(f'\nSkipping {fname} (real image issue: {real_diagnosis})')
            continue
        
        img_gen = load_image_with_fallback(gen_img_path)
        img_real = load_image_with_fallback(real_img_path)
        
        if img_gen is None:
            if verbose:
                print(f'\nSkipping {fname} (failed to load generated image: {gen_img_path})')
            continue
            
        if img_real is None:
            if verbose:
                print(f'\nSkipping {fname} (failed to load real image: {real_img_path})')
            continue
            
        # Resize to match
        if img_gen.shape != img_real.shape:
            img_gen = cv2.resize(img_gen, (img_real.shape[1], img_real.shape[0]))
            
        metrics = calculate_metrics(img_gen, img_real)
        psnr_list.append(metrics[0])
        ssim_list.append(metrics[1])
        mse_list.append(metrics[2])
        
        if metrics[3] is not None:
            lpips_list.append(metrics[3])
        
        # Calculate CLIP Score
        clip_score = calculate_clip_score(img_gen, img_real)
        if clip_score is not None:
            clip_list.append(clip_score)
        
        # Store images for FID calculation
        real_images_for_fid.append(img_real)
        gen_images_for_fid.append(img_gen)
            
        valid_files += 1
        
        # Update progress bar description with current stats
        if verbose and isinstance(pbar, tqdm):
            pbar.set_postfix({
                'valid': valid_files,
                'total': len(files),
                'psnr': f"{np.mean(psnr_list):.2f}" if psnr_list else "N/A",
                'clip': f"{np.mean(clip_list):.3f}" if clip_list else "N/A"
            })
    
    if verbose and isinstance(pbar, tqdm):
        pbar.close()
    
    # Calculate FID Score
    fid_score_val = None
    if len(real_images_for_fid) > 0 and not skip_fid:
        fid_score_val = calculate_fid_score(real_images_for_fid, gen_images_for_fid)
    elif skip_fid:
        if verbose:
            print(f"Skipping FID calculation for {model}/{category}")
    
    if valid_files == 0:
        return None
        
    results = {
        'model': model,
        'category': category,
        'total_files': len(files),
        'valid_files': valid_files,
        'psnr_mean': np.mean(psnr_list),
        'psnr_std': np.std(psnr_list),
        'ssim_mean': np.mean(ssim_list),
        'ssim_std': np.std(ssim_list),
        'mse_mean': np.mean(mse_list),
        'mse_std': np.std(mse_list),
        'lpips_mean': np.mean(lpips_list) if lpips_list else None,
        'lpips_std': np.std(lpips_list) if lpips_list else None,
        'lpips_valid_count': len(lpips_list),
        'clip_mean': np.mean(clip_list) if clip_list else None,
        'clip_std': np.std(clip_list) if clip_list else None,
        'clip_valid_count': len(clip_list),
        'fid_score': fid_score_val
    }
    
    return results

def run_benchmark(gen_folder='generated_images', real_folder='amazon_data', 
                  output_dir='benchmark_results', verbose=True, skip_fid=False):
    """Run comprehensive benchmark across all models and categories."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results storage
    all_results = []
    summary_results = []
    
    # Track progress
    total_combinations = len(MODELS) * len(CATEGORIES)
    current_combination = 0
    
    print(f"Starting benchmark with {len(MODELS)} models and {len(CATEGORIES)} categories")
    print(f"Total combinations to evaluate: {total_combinations}")
    print("=" * 80)
    
    # Create progress bar for overall benchmark
    overall_pbar = tqdm(total=total_combinations, desc="Overall Progress", unit="combination")
    
    for model in MODELS:
        model_results = []
        
        for category in CATEGORIES:
            current_combination += 1
            overall_pbar.set_description(f"Evaluating {model}/{category}")
            
            result = evaluate_model_category(model, category, gen_folder, real_folder, verbose, skip_fid)
            
            if result:
                all_results.append(result)
                model_results.append(result)
                
                if verbose:
                    lpips_str = f"{result['lpips_mean']:.4f}±{result['lpips_std']:.4f}" if result['lpips_mean'] else "N/A"
                    clip_str = f"{result['clip_mean']:.4f}±{result['clip_std']:.4f}" if result['clip_mean'] else "N/A"
                    fid_str = f"{result['fid_score']:.2f}" if result['fid_score'] else "N/A"
                    print(f"\nResults for {model}/{category}: PSNR={result['psnr_mean']:.2f}±{result['psnr_std']:.2f}, "
                          f"SSIM={result['ssim_mean']:.4f}±{result['ssim_std']:.4f}, "
                          f"LPIPS={lpips_str}, CLIP={clip_str}, FID={fid_str}")
            else:
                print(f"\nNo valid results for {model}/{category}")
            
            overall_pbar.update(1)
        
        # Calculate model summary
        if model_results:
            model_summary = {
                'model': model,
                'categories_evaluated': len(model_results),
                'psnr_mean': np.mean([r['psnr_mean'] for r in model_results]),
                'psnr_std': np.std([r['psnr_mean'] for r in model_results]),
                'ssim_mean': np.mean([r['ssim_mean'] for r in model_results]),
                'ssim_std': np.std([r['ssim_mean'] for r in model_results]),
                'mse_mean': np.mean([r['mse_mean'] for r in model_results]),
                'mse_std': np.std([r['mse_mean'] for r in model_results]),
                'lpips_mean': np.mean([r['lpips_mean'] for r in model_results if r['lpips_mean'] is not None]),
                'lpips_std': np.std([r['lpips_mean'] for r in model_results if r['lpips_mean'] is not None]),
                'clip_mean': np.mean([r['clip_mean'] for r in model_results if r['clip_mean'] is not None]),
                'clip_std': np.std([r['clip_mean'] for r in model_results if r['clip_mean'] is not None]),
                'fid_mean': np.mean([r['fid_score'] for r in model_results if r['fid_score'] is not None]),
                'fid_std': np.std([r['fid_score'] for r in model_results if r['fid_score'] is not None]),
                'total_files_processed': sum([r['valid_files'] for r in model_results])
            }
            summary_results.append(model_summary)
    
    overall_pbar.close()
    
    # Save detailed results
    if all_results:
        df_detailed = pd.DataFrame(all_results)
        detailed_csv_path = os.path.join(output_dir, f'benchmark_detailed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        df_detailed.to_csv(detailed_csv_path, index=False)
        print(f"\nDetailed results saved to: {detailed_csv_path}")
        
        # Save summary results
        df_summary = pd.DataFrame(summary_results)
        summary_csv_path = os.path.join(output_dir, f'benchmark_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        df_summary.to_csv(summary_csv_path, index=False)
        print(f"Summary results saved to: {summary_csv_path}")
        
        # Save JSON results
        json_path = os.path.join(output_dir, f'benchmark_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        json_data = {
            'detailed_results': convert_numpy_types(all_results),
            'summary_results': convert_numpy_types(summary_results),
            'metadata': {
                'models': MODELS,
                'categories': CATEGORIES,
                'timestamp': datetime.now().isoformat(),
                'total_combinations': total_combinations,
                'successful_evaluations': len(all_results)
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"JSON results saved to: {json_path}")
        
        # Print summary table
        print("\n" + "=" * 120)
        print("BENCHMARK SUMMARY")
        print("=" * 120)
        print(f"{'Model':<15} {'Categories':<10} {'PSNR':<12} {'SSIM':<12} {'LPIPS':<12} {'CLIP':<12} {'FID':<12} {'Files':<8}")
        print("-" * 120)
        
        for summary in summary_results:
            lpips_str = f"{summary['lpips_mean']:.4f}" if summary['lpips_mean'] is not None else "N/A"
            clip_str = f"{summary['clip_mean']:.4f}" if summary['clip_mean'] is not None else "N/A"
            fid_str = f"{summary['fid_mean']:.2f}" if summary['fid_mean'] is not None else "N/A"
            print(f"{summary['model']:<15} {summary['categories_evaluated']:<10} "
                  f"{summary['psnr_mean']:<12.2f} {summary['ssim_mean']:<12.4f} "
                  f"{lpips_str:<12} {clip_str:<12} {fid_str:<12} {summary['total_files_processed']:<8}")
        
        return df_detailed, df_summary
    
    else:
        print("No valid results found!")
        return None, None

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive benchmark for image generation models')
    parser.add_argument('--gen_folder', default='generated_images_bench', 
                       help='Folder containing generated images')
    parser.add_argument('--real_folder', default='amazon_data_rec', 
                       help='Folder containing real images')
    parser.add_argument('--output_dir', default='benchmark_results', 
                       help='Output directory for results')
    parser.add_argument('--quiet', action='store_true', 
                       help='Reduce verbose output')
    parser.add_argument('--skip-fid', action='store_true',
                       help='Skip FID score calculation (useful if FID causes errors)')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Run benchmark
    detailed_results, summary_results = run_benchmark(
        gen_folder=args.gen_folder,
        real_folder=args.real_folder,
        output_dir=args.output_dir,
        verbose=verbose,
        skip_fid=args.skip_fid
    )
    
    if detailed_results is not None:
        print(f"\nBenchmark completed successfully!")
        print(f"Results saved to: {args.output_dir}")
    else:
        print("Benchmark failed - no valid results generated")

if __name__ == '__main__':
    main() 