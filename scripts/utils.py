from pathlib import Path
import os
import numpy as np
from scipy import stats

project_root = Path(os.getcwd())

# Check if the selected basin-id was used for training the lumped LSTM
def is_train_basin(basin_id):
    with open(project_root / 'model' /  'basins' / 'train_basins.txt', 'r') as f:
        gl_basins = f.readlines()
    gl_basins = [id[:-1] for id in gl_basins]

    output = True
    if basin_id not in gl_basins:
        output = False
    
    return output


def kling_gupta_efficiency(predicted, observed):
    if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
        return np.asarray([np.nan])
    alpha = predicted.std() / observed.std()
    beta = predicted.mean() / observed.mean()
    r, p = stats.pearsonr(np.squeeze(predicted).astype(float), np.squeeze(observed).astype(float))
    return 1 - np.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)


def find_gauged_subbasin(basin_id, exp_name):
    routing_dir = Path(project_root / 'data' / f'{basin_id}_{exp_name}')
    with open(routing_dir / f'{basin_id}.rvh', 'r') as rvh:
        alllines = rvh.readlines()
    for i in range(len(alllines)):
        if ":SubBasins" in alllines[i]:
            start_index = i+3
        if ":EndSubBasins" in alllines[i]:
            end_index = i
            break
    for li in range(start_index, end_index):
        downstream_id = alllines[li].split()[2]
        if downstream_id == '-1':
            gauged_sub = alllines[li].split()[0]

    return gauged_sub
