
import os
import pandas as pd
import numpy as np
import re
import json
from datetime import datetime
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Suppress specific RoBERTa initialization warnings
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# Try to import BERTScore and BLEU using evaluate library
try:
    from evaluate import load
    bertscore = load("bertscore")
    bleu = load("bleu")
    BERTSCORE_AVAILABLE = True
    BLEU_AVAILABLE = True
except ImportError:
    print("Warning: BERTScore/BLEU not available. Install with: pip install evaluate")
    BERTSCORE_AVAILABLE = False
    BLEU_AVAILABLE = False

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Define all models and categories
# MODELS = [
#     'gemma_vl_4b',
#     'gemma_vl_12b', 
#     'gemma_vl_27b',
#     'qwen_vl_7b',
#     'qwen_vl_32b',
#     'qwen_vl_72b',
# ]

MODELS = [
    'qwen_vl_unknown'
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

def preprocess_text(text):
    """Preprocess text for similarity calculation."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def calculate_text_similarity_metrics(text1, text2):
    """Calculate multiple text similarity metrics including BERTScore and BLEU."""
    # Preprocess texts
    text1_clean = preprocess_text(text1)
    text2_clean = preprocess_text(text2)
    
    if not text1_clean or not text2_clean:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    # 1. TF-IDF Cosine Similarity
    try:
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )
        tfidf_matrix = vectorizer.fit_transform([text1_clean, text2_clean])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except:
        cosine_sim = 0.0
    
    # 2. TF-IDF Euclidean Distance (normalized)
    try:
        euclidean_dist = euclidean_distances(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        # Normalize to [0, 1] range (lower distance = higher similarity)
        euclidean_sim = 1.0 / (1.0 + euclidean_dist)
    except:
        euclidean_sim = 0.0
    
    # 3. Jaccard Similarity (word-based)
    try:
        words1 = set(text1_clean.split())
        words2 = set(text2_clean.split())
        if words1 and words2:
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            jaccard_sim = len(intersection) / len(union) if union else 0.0
        else:
            jaccard_sim = 0.0
    except:
        jaccard_sim = 0.0
    
    # 4. Word Overlap Ratio
    try:
        words1 = set(text1_clean.split())
        words2 = set(text2_clean.split())
        if words1 and words2:
            overlap = len(words1.intersection(words2))
            total_unique = len(words1.union(words2))
            overlap_ratio = overlap / total_unique if total_unique > 0 else 0.0
        else:
            overlap_ratio = 0.0
    except:
        overlap_ratio = 0.0
    
    # 5. BERTScore
    bertscore_sim = 0.0
    if BERTSCORE_AVAILABLE:
        try:
            # Use original texts for BERTScore (not preprocessed)
            if isinstance(text1, str) and isinstance(text2, str) and text1.strip() and text2.strip():
                results = bertscore.compute(predictions=[text1], references=[text2], lang="en")
                bertscore_sim = float(results['f1'][0])  # Use F1 score as similarity
            else:
                bertscore_sim = 0.0
        except Exception as e:
            print(f"BERTScore calculation error: {e}")
            bertscore_sim = 0.0
    else:
        bertscore_sim = 0.0
    
    # 6. BLEU Score
    bleu_score = 0.0
    if BLEU_AVAILABLE:
        try:
            # Use original texts for BLEU (not preprocessed)
            if isinstance(text1, str) and isinstance(text2, str) and text1.strip() and text2.strip():
                # For BLEU, we treat the generated text as prediction and real text as reference
                # We use a single reference (the real text) for each prediction
                results = bleu.compute(predictions=[text1], references=[[text2]])
                bleu_score = float(results['bleu'])
            else:
                bleu_score = 0.0
        except Exception as e:
            print(f"BLEU calculation error: {e}")
            bleu_score = 0.0
    else:
        bleu_score = 0.0
    
    return cosine_sim, euclidean_sim, jaccard_sim, overlap_ratio, bertscore_sim, bleu_score

def load_real_data(category, data_dir='amazon_data'):
    """Load real text data from Amazon dataset and validate ID count."""
    data_file = os.path.join(data_dir, category, f'{category}_data.txt')
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Real data file not found: {data_file}")
    
    real_data = {}
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    item_id = parts[0].strip()
                    content = parts[1].strip()
                    real_data[item_id] = content
    except Exception as e:
        raise Exception(f"Error reading real data file {data_file}: {e}")
    
    # Validate that we have 1000 IDs for each category
    expected_count = 1000
    actual_count = len(real_data)
    if actual_count != expected_count:
        raise ValueError(f"Category {category}: Expected {expected_count} IDs, but found {actual_count} IDs in {data_file}")
    
    print(f"Category {category}: Successfully loaded {actual_count} IDs from {data_file}")
    return real_data

def load_generated_data(model, category, output_dir='organized_outputs'):
    """Load generated text data from model outputs."""
    # Try different file patterns
    file_patterns = [
        f'{category}_gpu*.tsv',
        f'{category}_gpu*.csv',
        f'{category}.tsv',
        f'{category}.csv'
    ]
    
    titles_dir = os.path.join(output_dir, model, 'titles')
    if not os.path.exists(titles_dir):
        raise FileNotFoundError(f"Titles directory not found: {titles_dir}")
    
    generated_data = {}
    
    for pattern in file_patterns:
        matching_files = []
        for file in os.listdir(titles_dir):
            if re.match(pattern.replace('*', '.*'), file):
                matching_files.append(file)
        
        if matching_files:
            # Use the first matching file
            file_path = os.path.join(titles_dir, matching_files[0])
            try:
                # Try to read as TSV first
                try:
                    df = pd.read_csv(file_path, sep='\t')
                except:
                    # If TSV fails, try CSV
                    df = pd.read_csv(file_path)
                
                # Check if we have the expected columns
                if len(df.columns) >= 2:
                    for _, row in df.iterrows():
                        item_id = str(row.iloc[0]).strip()
                        title = str(row.iloc[1]).strip()
                        if item_id and title and item_id != 'nan' and title != 'nan':
                            generated_data[item_id] = title
                
                print(f"Loaded {len(generated_data)} items from {file_path}")
                break
                
            except Exception as e:
                print(f"Error reading generated data file {file_path}: {e}")
                continue
    
    if not generated_data:
        raise FileNotFoundError(f"No generated data found for {model}/{category} in {titles_dir}")
    
    return generated_data

def evaluate_model_category_text(model, category, real_data_dir='amazon_data', 
                               output_dir='organized_outputs', verbose=True):
    """Evaluate text similarity for a specific model on a specific category."""
    
    # Load real data (this will validate 1000 IDs)
    real_data = load_real_data(category, real_data_dir)
    
    # Load generated data
    generated_data = load_generated_data(model, category, output_dir)
    
    # Check for missing IDs in generated data
    missing_ids = set(real_data.keys()) - set(generated_data.keys())
    if missing_ids:
        missing_count = len(missing_ids)
        total_expected = len(real_data)
        raise ValueError(f"Missing {missing_count} IDs in generated data for {model}/{category}. "
                       f"Expected {total_expected} IDs, found {len(generated_data)} IDs. "
                       f"Missing IDs: {list(missing_ids)[:10]}{'...' if missing_count > 10 else ''}")
    
    # Use all real data IDs (should be 1000)
    common_ids = list(real_data.keys())
    
    if verbose:
        print(f"Evaluating {len(common_ids)} items for {model}/{category}")
    
    # Calculate metrics for each pair
    cosine_sims, euclidean_sims, jaccard_sims, overlap_ratios, bertscore_sims, bleu_scores = [], [], [], [], [], []
    valid_pairs = 0
    
    for item_id in common_ids:
        real_text = real_data[item_id]
        generated_text = generated_data[item_id]
        
        # Calculate similarity metrics
        cosine_sim, euclidean_sim, jaccard_sim, overlap_ratio, bertscore_sim, bleu_score = calculate_text_similarity_metrics(
            real_text, generated_text
        )
        
        cosine_sims.append(cosine_sim)
        euclidean_sims.append(euclidean_sim)
        jaccard_sims.append(jaccard_sim)
        overlap_ratios.append(overlap_ratio)
        bertscore_sims.append(bertscore_sim)
        bleu_scores.append(bleu_score)
        valid_pairs += 1
        
        if verbose and valid_pairs % 100 == 0:
            print(f"  Processed {valid_pairs}/{len(common_ids)} pairs for {model}/{category}")
    
    if valid_pairs == 0:
        raise ValueError(f"No valid pairs found for {model}/{category}")
    
    # Calculate statistics
    results = {
        'model': model,
        'category': category,
        'total_items': len(common_ids),
        'valid_pairs': valid_pairs,
        'cosine_sim_mean': np.mean(cosine_sims),
        'cosine_sim_std': np.std(cosine_sims),
        'euclidean_sim_mean': np.mean(euclidean_sims),
        'euclidean_sim_std': np.std(euclidean_sims),
        'jaccard_sim_mean': np.mean(jaccard_sims),
        'jaccard_sim_std': np.std(jaccard_sims),
        'overlap_ratio_mean': np.mean(overlap_ratios),
        'overlap_ratio_std': np.std(overlap_ratios),
        'bertscore_sim_mean': np.mean(bertscore_sims),
        'bertscore_sim_std': np.std(bertscore_sims),
        'bleu_score_mean': np.mean(bleu_scores),
        'bleu_score_std': np.std(bleu_scores)
    }
    
    return results

def run_text_benchmark(real_data_dir='amazon_data', output_dir='organized_outputs',
                      benchmark_output_dir='text_benchmark_results', verbose=True):
    """Run comprehensive text benchmark across all models and categories."""
    
    # Create output directory
    os.makedirs(benchmark_output_dir, exist_ok=True)
    
    # Initialize results storage
    all_results = []
    summary_results = []
    failed_combinations = []
    
    # Track progress
    total_combinations = len(MODELS) * len(CATEGORIES)
    current_combination = 0
    
    print(f"Starting text benchmark with {len(MODELS)} models and {len(CATEGORIES)} categories")
    print(f"Total combinations to evaluate: {total_combinations}")
    print("=" * 80)
    
    for model in MODELS:
        model_results = []
        
        for category in CATEGORIES:
            current_combination += 1
            print(f"\n[{current_combination}/{total_combinations}] Evaluating {model} on {category}")
            
            try:
                result = evaluate_model_category_text(
                    model, category, real_data_dir, output_dir, verbose
                )
                
                all_results.append(result)
                model_results.append(result)
                
                if verbose:
                    print(f"  Results: Cosine={result['cosine_sim_mean']:.4f}±{result['cosine_sim_std']:.4f}, "
                          f"Jaccard={result['jaccard_sim_mean']:.4f}±{result['jaccard_sim_std']:.4f}, "
                          f"Overlap={result['overlap_ratio_mean']:.4f}±{result['overlap_ratio_std']:.4f}, "
                          f"BERTScore={result['bertscore_sim_mean']:.4f}±{result['bertscore_sim_std']:.4f}, "
                          f"BLEU={result['bleu_score_mean']:.4f}±{result['bleu_score_std']:.4f}")
                          
            except Exception as e:
                error_msg = f"Error evaluating {model}/{category}: {str(e)}"
                print(f"  ERROR: {error_msg}")
                failed_combinations.append({
                    'model': model,
                    'category': category,
                    'error': str(e)
                })
        
        # Calculate model summary
        if model_results:
            model_summary = {
                'model': model,
                'categories_evaluated': len(model_results),
                'cosine_sim_mean': np.mean([r['cosine_sim_mean'] for r in model_results]),
                'cosine_sim_std': np.std([r['cosine_sim_mean'] for r in model_results]),
                'euclidean_sim_mean': np.mean([r['euclidean_sim_mean'] for r in model_results]),
                'euclidean_sim_std': np.std([r['euclidean_sim_mean'] for r in model_results]),
                'jaccard_sim_mean': np.mean([r['jaccard_sim_mean'] for r in model_results]),
                'jaccard_sim_std': np.std([r['jaccard_sim_mean'] for r in model_results]),
                'overlap_ratio_mean': np.mean([r['overlap_ratio_mean'] for r in model_results]),
                'overlap_ratio_std': np.std([r['overlap_ratio_mean'] for r in model_results]),
                'bertscore_sim_mean': np.mean([r['bertscore_sim_mean'] for r in model_results]),
                'bertscore_sim_std': np.std([r['bertscore_sim_mean'] for r in model_results]),
                'bleu_score_mean': np.mean([r['bleu_score_mean'] for r in model_results]),
                'bleu_score_std': np.std([r['bleu_score_mean'] for r in model_results]),
                'total_items_processed': sum([r['valid_pairs'] for r in model_results])
            }
            summary_results.append(model_summary)
    
    # Save detailed results
    if all_results:
        df_detailed = pd.DataFrame(all_results)
        detailed_csv_path = os.path.join(benchmark_output_dir, f'text_benchmark_detailed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        df_detailed.to_csv(detailed_csv_path, index=False)
        print(f"\nDetailed results saved to: {detailed_csv_path}")
        
        # Save summary results
        df_summary = pd.DataFrame(summary_results)
        summary_csv_path = os.path.join(benchmark_output_dir, f'text_benchmark_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        df_summary.to_csv(summary_csv_path, index=False)
        print(f"Summary results saved to: {summary_csv_path}")
        
        # Save failed combinations
        if failed_combinations:
            failed_csv_path = os.path.join(benchmark_output_dir, f'text_benchmark_failed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            df_failed = pd.DataFrame(failed_combinations)
            df_failed.to_csv(failed_csv_path, index=False)
            print(f"Failed combinations saved to: {failed_csv_path}")
        
        # Save JSON results
        json_path = os.path.join(benchmark_output_dir, f'text_benchmark_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        
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
            'failed_combinations': failed_combinations,
            'metadata': {
                'models': MODELS,
                'categories': CATEGORIES,
                'timestamp': datetime.now().isoformat(),
                'total_combinations': total_combinations,
                'successful_evaluations': len(all_results),
                'failed_evaluations': len(failed_combinations)
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"JSON results saved to: {json_path}")
        
        # Print summary table
        print("\n" + "=" * 100)
        print("TEXT BENCHMARK SUMMARY")
        print("=" * 100)
        print(f"{'Model':<15} {'Categories':<10} {'Cosine':<12} {'Jaccard':<12} {'Overlap':<12} {'BERTScore':<12} {'BLEU':<12} {'Items':<8}")
        print("-" * 120)
        
        for summary in summary_results:
            print(f"{summary['model']:<15} {summary['categories_evaluated']:<10} "
                  f"{summary['cosine_sim_mean']:<12.4f} {summary['jaccard_sim_mean']:<12.4f} "
                  f"{summary['overlap_ratio_mean']:<12.4f} {summary['bertscore_sim_mean']:<12.4f} "
                  f"{summary['bleu_score_mean']:<12.4f} {summary['total_items_processed']:<8}")
        
        # Print failed combinations summary
        if failed_combinations:
            print(f"\nFailed combinations ({len(failed_combinations)}):")
            for failed in failed_combinations:
                print(f"  {failed['model']}/{failed['category']}: {failed['error']}")
        
        return df_detailed, df_summary
    
    else:
        print("No valid results found!")
        return None, None

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive text benchmark for text generation models')
    parser.add_argument('--real_data_dir', default='amazon_data_bench', 
                       help='Directory containing real text data')
    parser.add_argument('--output_dir', default='organized_outputs_bench', 
                       help='Directory containing generated text outputs')
    parser.add_argument('--benchmark_output_dir', default='text_benchmark_results', 
                       help='Output directory for benchmark results')
    parser.add_argument('--quiet', action='store_true', 
                       help='Reduce verbose output')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Run benchmark
    detailed_results, summary_results = run_text_benchmark(
        real_data_dir=args.real_data_dir,
        output_dir=args.output_dir,
        benchmark_output_dir=args.benchmark_output_dir,
        verbose=verbose
    )
    
    if detailed_results is not None:
        print(f"\nText benchmark completed successfully!")
        print(f"Results saved to: {args.benchmark_output_dir}")
    else:
        print("Text benchmark failed - no valid results generated")

if __name__ == '__main__':
    main() 