# MMPCBench: Missing Modality Product Completion Benchmark

A comprehensive benchmark for evaluating MLLM-based missing modality product completion using Amazon product data. This project provides tools for downloading, processing, and evaluating multimodal recommendation systems with missing data scenarios.

## Project Overview

MMPCBench consists of two main components:
- **CQBench**: For multimodal completion quality evaluation
- **Recbench**: For recommendation system evaluation with missing data

## Quick Start

### 1. Data Download and Processing

First, download and process the Amazon dataset:

```bash
cd MMPCBench
python download_amazon_data_rec.py
```

This script will:
- Download Amazon product data for 9 categories
- Process and filter the data
- Generate multimodal datasets with images and text
- Save in MSRBench format for further processing

### 2. Generate Missing Data Completions

#### Using CQBench (Completion Quality)

```bash
cd MMPCBench/CQBench
python imagine_amazon_data_bench.py --model qwen2.5-vl-7b --gpu_id 0 --mode title
```

This generates product titles from images or image prompts from descriptions.

#### Using Recbench (Recommendation Completion)

```bash
cd MMPCBench/Recbench
python imagine_amazon_data_rec.py --model qwen2.5-vl-7b --gpu_id 0 --mode title
```

This generates completions specifically for recommendation scenarios.

### 3. Run Recommendation Experiments

```bash
cd MMPCBench/Recbench/src
python main.py -m FREEDOM -d all_beauty --completion_method qwen_vl_7b --missing_type image
```

## Project Structure

```
MMPCBench/
├── download_amazon_data_rec.py    # Data download and processing
├── CQBench/                       # Completion quality evaluation
│   ├── imagine_amazon_data_bench.py
│   ├── generate_images_from_prompts.py
│   └── gather_text.py
├── Recbench/                      # Recommendation system evaluation
│   ├── imagine_amazon_data_rec.py
│   ├── run_my_experiments.py
│   ├── src/
│   │   ├── main.py               # Main experiment runner
│   │   ├── models/               # Model implementations
│   │   └── utils/                # Utility functions
│   └── usage_instructions.md     # Usage instructions
└── README.md                      # This file
```

## Supported Models

### Vision-Language Models for Completion
- **Qwen2.5-VL**: 7B, 32B, 72B variants
- **Gemma-3**: 4B, 12B, 27B variants  
- **LLaVA**: Multiple variants

### Recommendation Models
- **VBPR**
- **BM3**
- **FREEDOM**
- ... (and all available models from MMRec)

## Supported Datasets

Amazon product categories:
- All Beauty
- Toys and Games
- Video Games
- Home and Kitchen
- Electronics
- Industrial and Scientific
- Office Products
- Musical Instruments
- Arts, Crafts and Sewing

## Usage Examples

### Batch Experiment Running

```bash
# Run all experiments on all datasets
cd MMPCBench/Recbench
python run_my_experiments.py
```

### Custom Configuration

```bash
# Run specific model on specific dataset
cd MMPCBench/Recbench/src
python main.py -m FREEDOM -d electronics --completion_method qwen_vl_7b --missing_type image
```

## Configuration

### GPU Settings
Edit `run_my_experiments.py` to set your GPU ID:
```python
GPU_ID = 0  # Change to your GPU ID
```

### Experiment Parameters
- `missing_rate`: Percentage of missing data (0.0-1.0)
- `completion_method`: Method for completing missing data
- `missing_type`: Type of missing data (image/text)

## Dependencies

### Core Requirements
```bash
pip install torch torchvision
pip install transformers accelerate
pip install pandas numpy tqdm
pip install psutil GPUtil
```

### Vision-Language Models
```bash
# For Qwen models
pip install qwen-vl-utils[decord]

# For Gemma models
pip install transformers[torch]

# For InternVL3
pip install lmdeploy

# For LLaVA
pip install transformers[torch]
```
