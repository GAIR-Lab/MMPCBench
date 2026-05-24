#!/usr/bin/env python3
"""
MMRec FREEDOM model text missing experiment running script
Specifically for FREEDOM model, text missing, missing_rate=1.0 experiments
Supports multi-GPU parallel execution, each GPU runs 10 processes simultaneously
"""

import os
import subprocess
import time
import argparse
import multiprocessing as mp
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import signal
import sys
import json
try:
    import psutil
except ImportError:
    print("❌ Error: psutil package required")
    print("Please run: pip install psutil")
    sys.exit(1)
import atexit

# Model configuration - only use FREEDOM
MODELS = ["VBPR","FREEDOM","BM3"]

# Dataset configuration - all datasets
DATASETS = [
    'electronics_small',
    'home_and_kitchen_small',
    'all_beauty',
    'toys_and_games', 
    'video_games',
    'industrial_and_scientific',
    'office_products',
    'musical_instruments',
    'arts_crafts_and_sewing'
]

EXPERIMENTS = [
    # === Text missing experiments - missing_rate=1.0 ===
    {'missing_rate': 1.0, 'completion_method': 'random', 'missing_type': 'text', 'name': 'text_missing_100_random'},
    {'missing_rate': 1.0, 'completion_method': 'zeros', 'missing_type': 'text', 'name': 'text_missing_100_zeros'},
    {'missing_rate': 1.0, 'completion_method': 'random', 'missing_type': 'image', 'name': 'image_missing_100_random'},
    {'missing_rate': 1.0, 'completion_method': 'zeros', 'missing_type': 'image', 'name': 'image_missing_100_zeros'},
]

# Global variables for tracking progress
progress_lock = threading.Lock()
completed_experiments = 0
total_experiments = 0
failed_experiments = []
success_experiments = []

# Global variables for process management
executor = None
emergency_stop = False
emergency_stop_lock = threading.Lock()

# Exception monitoring configuration
MAX_MEMORY_USAGE = 0.9  # 90% memory usage threshold
MAX_GPU_MEMORY_USAGE = 0.95  # 95% GPU memory usage threshold

# Monitor thread control
monitor_thread = None
monitor_stop = False
monitor_stop_lock = threading.Lock()

def signal_handler(signum, frame):
    """Handle interrupt signal"""
    print("\n\nReceived interrupt signal, gracefully exiting...")
    emergency_terminate_all()
    sys.exit(0)

def emergency_terminate_all():
    """Emergency terminate all related processes"""
    global emergency_stop, executor
    
    with emergency_stop_lock:
        if emergency_stop:
            return
        emergency_stop = True
    
    print("\n🚨 Detected exception, emergency terminating all processes...")
    
    # Terminate process pool
    if executor:
        try:
            executor.shutdown(wait=False)
            print("✅ Process pool closed")
        except Exception as e:
            print(f"⚠️ Error closing process pool: {e}")
    
    # Find and terminate all related Python processes
    current_pid = os.getpid()
    killed_count = 0
    
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # Check if it's a related Python process (but not current process)
                if (proc.info['name'] and 'python' in proc.info['name'].lower() and 
                    proc.info['pid'] != current_pid and 
                    proc.info['cmdline']):
                    
                    cmdline = ' '.join(proc.info['cmdline'])
                    # Check if it's our experiment process
                    if ('main.py' in cmdline or 'run_freedom_text_experiments.py' in cmdline):
                        print(f"🔄 Terminating process {proc.info['pid']}: {cmdline[:100]}...")
                        proc.terminate()
                        killed_count += 1
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        
        print(f"✅ Terminated {killed_count} related processes")
        
        # Wait for processes to fully terminate
        time.sleep(2)
        
        # Force kill still running processes
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if (proc.info['name'] and 'python' in proc.info['name'].lower() and 
                    proc.info['pid'] != current_pid and 
                    proc.info['cmdline']):
                    
                    cmdline = ' '.join(proc.info['cmdline'])
                    if ('main.py' in cmdline or 'run_freedom_text_experiments.py' in cmdline):
                        print(f"💀 Force terminating process {proc.info['pid']}")
                        proc.kill()
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
                
    except Exception as e:
        print(f"⚠️ Error terminating processes: {e}")
    
    print("🚨 Emergency termination completed")

def cleanup_on_exit():
    """Cleanup function when program exits"""
    if not emergency_stop:
        emergency_terminate_all()

# Register cleanup function on exit
atexit.register(cleanup_on_exit)

def check_system_resources():
    """Check system resource usage"""
    try:
        # Check memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent / 100
        
        if memory_usage > MAX_MEMORY_USAGE:
            print(f"🚨 Memory usage too high: {memory_usage:.1%} > {MAX_MEMORY_USAGE:.1%}")
            return False
        
        # Check GPU usage (if available)
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_usage = info.used / info.total
                
                if gpu_usage > MAX_GPU_MEMORY_USAGE:
                    print(f"🚨 GPU {i} memory usage too high: {gpu_usage:.1%} > {MAX_GPU_MEMORY_USAGE:.1%}")
                    return False
                    
        except ImportError:
            # pynvml not available, skip GPU check
            pass
        except Exception as e:
            print(f"⚠️ GPU monitoring error: {e}")
        
        return True
        
    except Exception as e:
        print(f"⚠️ System resource check error: {e}")
        return True  # Don't block execution on error

def resource_monitor():
    """Resource monitoring thread"""
    global monitor_stop
    
    while True:
        with monitor_stop_lock:
            if monitor_stop:
                break
        
        try:
            if not check_system_resources():
                print("🚨 Monitor thread detected system resource exception, triggering emergency stop")
                emergency_terminate_all()
                break
            
            # Check every 30 seconds
            time.sleep(30)
            
        except Exception as e:
            print(f"⚠️ Monitor thread error: {e}")
            time.sleep(30)

def start_resource_monitor():
    """Start resource monitoring thread"""
    global monitor_thread, monitor_stop
    
    with monitor_stop_lock:
        monitor_stop = False
    
    monitor_thread = threading.Thread(target=resource_monitor, daemon=True)
    monitor_thread.start()
    print("🔍 Resource monitoring thread started")

def stop_resource_monitor():
    """Stop resource monitoring thread"""
    global monitor_stop
    
    with monitor_stop_lock:
        monitor_stop = True
    
    if monitor_thread and monitor_thread.is_alive():
        monitor_thread.join(timeout=5)
        print("🔍 Resource monitoring thread stopped")

def run_single_experiment(experiment):
    """Run single experiment function - for multi-process"""
    global emergency_stop
    
    # Check if emergency stop has been triggered
    with emergency_stop_lock:
        if emergency_stop:
            return {
                'exp_id': experiment.get('exp_id', 'unknown'),
                'status': 'CANCELLED',
                'message': 'Experiment emergency terminated',
                'duration': 0,
                'gpu_id': experiment.get('gpu_id', 0)
            }
    
    model = experiment['model']
    dataset = experiment['dataset']
    gpu_id = experiment['gpu_id']
    missing_rate = experiment['missing_rate']
    completion_method = experiment['completion_method']
    missing_type = experiment['missing_type']
    exp_name = experiment['exp_name']
    
    # Build command arguments
    cmd_parts = [
        "python main.py",
        f"-m {model}",
        f"-d {dataset}",
        f"--gpu_id {gpu_id}",
        f"--missing_rate {missing_rate}",
        f"--completion_method {completion_method}",
        f"--missing_type {missing_type}"
    ]
    
    cmd = " ".join(cmd_parts)
    
    # Generate experiment identifier
    exp_id = f"{model}_{dataset}_{exp_name}" if exp_name else f"{model}_{dataset}_missing_{missing_rate}_{completion_method}_{missing_type}"
    
    # Record start time
    start_time = time.time()
    
    try:
        # Check system resources before running experiment
        if not check_system_resources():
            print(f"🚨 Insufficient system resources, skipping experiment: {exp_id}")
            emergency_terminate_all()
            return {
                'exp_id': exp_id,
                'status': 'RESOURCE_ERROR',
                'message': f'Insufficient system resources: {exp_id}',
                'duration': 0,
                'gpu_id': gpu_id
            }
        
        # Run experiment, timeout 2 hours
        result = subprocess.run(
            cmd,
            shell=True,
            timeout=22000,
            capture_output=True,
            text=True
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            status = "SUCCESS"
            message = f"✅ Success: {exp_id} (Duration: {duration:.1f}s)"
        else:
            status = "FAILED"
            message = f"❌ Failed: {exp_id} (Duration: {duration:.1f}s) - {result.stderr}"
            
    except subprocess.TimeoutExpired:
        end_time = time.time()
        duration = end_time - start_time
        status = "TIMEOUT"
        message = f"⏰ Timeout: {exp_id} (Duration: {duration:.1f}s)"
        
        # Trigger emergency stop
        print(f"\n🚨 Detected timeout exception: {exp_id}")
        emergency_terminate_all()
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        status = "ERROR"
        message = f"💥 Error: {exp_id} (Duration: {duration:.1f}s) - {e}"
        
        # Trigger emergency stop
        print(f"\n🚨 Detected exception: {exp_id} - {e}")
        emergency_terminate_all()
    
    return {
        'exp_id': exp_id,
        'status': status,
        'message': message,
        'duration': duration,
        'gpu_id': gpu_id
    }

def update_progress(result):
    """Update progress and results"""
    global completed_experiments, failed_experiments, success_experiments, emergency_stop
    
    with progress_lock:
        completed_experiments += 1
        
        if result['status'] == 'SUCCESS':
            success_experiments.append(result)
        else:
            failed_experiments.append(result)
        
        # Print progress
        progress = (completed_experiments / total_experiments) * 100
        print(f"\n[{completed_experiments}/{total_experiments}] {progress:.1f}% - {result['message']}")
        
        # Check if emergency stop is needed
        if result['status'] in ['TIMEOUT', 'ERROR']:
            print(f"\n🚨 Detected exception status: {result['status']}")
            emergency_terminate_all()
            return
        
        # Print GPU usage
        gpu_usage = {}
        for exp in success_experiments + failed_experiments:
            gpu = exp['gpu_id']
            gpu_usage[gpu] = gpu_usage.get(gpu, 0) + 1
        
        if gpu_usage:
            gpu_info = ", ".join([f"GPU{gpu}: {count}" for gpu, count in gpu_usage.items()])
            print(f"GPU usage: {gpu_info}")

def generate_all_experiments(gpu_ids):
    """Generate all experiment combinations"""
    all_experiments = []
    
    for model in MODELS:
        for dataset in DATASETS:
            for i, exp_config in enumerate(EXPERIMENTS):
                # Round-robin GPU allocation
                gpu_id = gpu_ids[i % len(gpu_ids)]
                
                experiment = {
                    'model': model,
                    'dataset': dataset,
                    'gpu_id': gpu_id,
                    'missing_rate': exp_config['missing_rate'],
                    'completion_method': exp_config['completion_method'],
                    'missing_type': exp_config['missing_type'],
                    'exp_name': exp_config.get('name', '')
                }
                all_experiments.append(experiment)
    
    return all_experiments

def print_experiment_summary(gpu_ids, max_workers):
    """Print experiment summary"""
    print("\n" + "="*80)
    print("FREEDOM Model Text Missing Experiment Configuration Summary")
    print("="*80)
    print(f"GPU IDs: {gpu_ids}")
    print(f"Maximum parallel processes: {max_workers}")
    print(f"Model: {MODELS[0]}")
    print(f"Number of datasets: {len(DATASETS)}")
    print(f"Number of experiment configurations: {len(EXPERIMENTS)}")
    
    print(f"\nDataset list:")
    for i, dataset in enumerate(DATASETS, 1):
        print(f"  {i:2d}. {dataset}")
    
    print(f"\nExperiment configuration details:")
    for i, exp in enumerate(EXPERIMENTS):
        name = exp.get('name', f'config_{i+1}')
        print(f"  {i+1:2d}. {name:30s} | missing_rate={exp['missing_rate']:4.1f} | "
              f"completion_method={exp['completion_method']:12s} | missing_type={exp['missing_type']:5s}")
    
    total_experiments = len(MODELS) * len(DATASETS) * len(EXPERIMENTS)
    estimated_time = total_experiments * 2 / 3600 / max_workers  # Assume 2 hours per experiment, considering parallelism
    
    print(f"\nTotal experiments: {total_experiments}")
    print(f"Estimated total time: {estimated_time:.1f} hours (parallel execution)")
    print("="*80)

def print_final_summary():
    """Print final summary"""
    print("\n" + "="*80)
    print("Experiment Completion Summary")
    print("="*80)
    print(f"Total experiments: {total_experiments}")
    print(f"Successful experiments: {len(success_experiments)}")
    print(f"Failed experiments: {len(failed_experiments)}")
    print(f"Success rate: {(len(success_experiments) / total_experiments) * 100:.1f}%")
    
    if success_experiments:
        avg_duration = sum(exp['duration'] for exp in success_experiments) / len(success_experiments)
        print(f"Average successful experiment duration: {avg_duration:.1f} seconds")
    
    if failed_experiments:
        print(f"\nFailed experiment list:")
        for exp in failed_experiments:
            print(f"  - {exp['exp_id']}: {exp['status']}")
    
    # Save results to file
    results = {
        'timestamp': datetime.now().isoformat(),
        'model': 'FREEDOM',
        'missing_type': 'text',
        'missing_rate': 1.0,
        'total_experiments': total_experiments,
        'success_experiments': len(success_experiments),
        'failed_experiments': len(failed_experiments),
        'success_rate': (len(success_experiments) / total_experiments) * 100,
        'success_details': success_experiments,
        'failed_details': failed_experiments
    }
    
    filename = f'freedom_text_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {filename}")
    print("="*80)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='FREEDOM Model Text Missing Experiment Running Script')
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0, 1], 
                       help='GPU device ids to use (default: 0 1)')
    parser.add_argument('--max_workers', type=int, default=10, 
                       help='Maximum number of parallel processes per GPU (default: 10)')
    return parser.parse_args()

def main():
    global executor
    
    # Set signal handling
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse command line arguments
    args = parse_args()
    
    # Calculate total parallel processes
    total_workers = len(args.gpu_ids) * args.max_workers
    
    global total_experiments
    total_experiments = len(MODELS) * len(DATASETS) * len(EXPERIMENTS)
    
    print("🎯 FREEDOM Model Text Missing Experiment Running Script")
    print(f"Model: {MODELS[0]}")
    print(f"Missing type: text")
    print(f"Missing rate: 1.0")
    print(f"Number of GPUs: {len(args.gpu_ids)}")
    print(f"Processes per GPU: {args.max_workers}")
    print(f"Total parallel processes: {total_workers}")
    
    # Print experiment summary
    print_experiment_summary(args.gpu_ids, total_workers)
    
    # Confirm whether to continue
    response = input("\nContinue running experiments? (y/n): ")
    if response.lower() != 'y':
        print("Experiments cancelled")
        return
    
    # Generate all experiments
    all_experiments = generate_all_experiments(args.gpu_ids)
    
    print(f"\nStarting {total_experiments} experiments with {total_workers} parallel processes...")
    print(f"GPU allocation: {args.gpu_ids}")
    print("⚠️ Note: Once timeout or exception is detected, all processes will be terminated immediately")
    
    # Start resource monitoring thread
    start_resource_monitor()
    
    start_time = time.time()
    
    try:
        # Use process pool to run experiments
        with ProcessPoolExecutor(max_workers=total_workers) as executor:
            # Submit all tasks
            future_to_experiment = {
                executor.submit(run_single_experiment, exp): exp 
                for exp in all_experiments
            }
            
            # Handle completed tasks
            for future in as_completed(future_to_experiment):
                try:
                    result = future.result()
                    update_progress(result)
                    
                    # Check if emergency stop has been triggered
                    with emergency_stop_lock:
                        if emergency_stop:
                            print("\n🚨 Detected emergency stop signal, exiting...")
                            break
                            
                except Exception as e:
                    print(f"Process execution exception: {e}")
                    emergency_terminate_all()
                    break
                    
    except KeyboardInterrupt:
        print("\n\nReceived keyboard interrupt signal")
        emergency_terminate_all()
    except Exception as e:
        print(f"\nProgram execution exception: {e}")
        emergency_terminate_all()
    finally:
        # Stop resource monitoring thread
        stop_resource_monitor()
    
    end_time = time.time()
    duration = (end_time - start_time) / 3600  # Hours
    
    # Check if emergency terminated
    with emergency_stop_lock:
        if emergency_stop:
            print("\n🚨 Experiments emergency terminated")
            print("Failure reason: Detected timeout or exception")
        else:
            # Print final summary
            print_final_summary()
            print(f"Total duration: {duration:.2f} hours")

if __name__ == '__main__':
    # Set multi-process start method
    mp.set_start_method('spawn', force=True)
    main()

'''

python run_freedom_text_experiments.py --gpu_ids 2 3 --max_workers 27
'''