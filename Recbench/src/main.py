# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SELFCFED_LGN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--gpu_id', type=int, default=2, help='GPU device id to use')
    parser.add_argument('--missing_rate', type=float, default=0.0, help='missing rate (0.0-1.0)')
    parser.add_argument('--completion_method', type=str, default='no_missing', 
                       choices=['no_missing', 'zeros', 'random', 'qwen_vl_7b','qwen_vl_32b','qwen_vl_72b','gemma_vl_4b','gemma_vl_12b','gemma_vl_27b'], 
                       help='completion method for missing features')
    parser.add_argument('--missing_type', type=str, default='none', 
                       choices=['none', 'image', 'text'], 
                       help='type of missing features')

    args, _ = parser.parse_known_args()

    config_dict = {
        'gpu_id': args.gpu_id,
        'missing_rate': args.missing_rate,
        'completion_method': args.completion_method,
        'missing_type': args.missing_type,
    }

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)


