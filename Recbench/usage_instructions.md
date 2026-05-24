# MMRec Batch Experiment Running Instructions

## Quick Start

### 1. Modify GPU Settings
Edit the `run_my_experiments.py` file, modify the GPU_ID on line 12:
```python
GPU_ID = 0  # Change to your GPU ID
```

### 2. Run Experiments
```bash
cd MMRec
python run_my_experiments.py
```

## Experiment Details

### Models (3)
- VBPR
- MMGCN  
- FREEDOM

### Datasets (9 Amazon Categories)
- all_beauty
- toys_and_games
- video_games
- home_and_kitchen
- electronics
- industrial_and_scientific
- office_products
- musical_instruments
- arts_crafts_and_sewing

### Experiment Order
The script will run 27 experiments in the following order:
1. VBPR on all_beauty
2. VBPR on toys_and_games
3. ...
4. MMGCN on all_beauty
5. ...
6. FREEDOM on arts_crafts_and_sewing

## Monitor Experiments

### View Progress
```bash
# Real-time GPU usage monitoring
watch -n 1 nvidia-smi

# View training logs
tail -f src/log/latest.log
```

### Stop Experiments
Press `Ctrl+C` to stop the current experiment, the script will continue with the next experiment.

## Notes

1. **Time Estimation**: Each experiment takes about 30 minutes to 2 hours, total may require 12-24 hours
2. **GPU Memory**: Ensure sufficient GPU memory
3. **Disk Space**: Ensure sufficient space to save models and logs
4. **Datasets**: Ensure all datasets are prepared

## Other Scripts

- `run_simple.py`: English version, same functionality
- `run_experiments.sh`: Bash script version
- `run_all_experiments.py`: Most complete version

## Troubleshooting

If an experiment fails, the script will continue with the next experiment. After all experiments complete, check the output to see which experiments succeeded/failed. 