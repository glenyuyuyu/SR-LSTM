from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import functools
import multiprocessing as mp
import os, sys
from tqdm import *
from utils import *
import pickle
import warnings
warnings.filterwarnings("ignore")

basin_id = str(sys.argv[1])
exp_name = str(sys.argv[2])

project_root = Path(sys.argv[3])
data_root = Path(project_root / 'data')
lstm_root = Path(project_root / 'model')

start_date = str(sys.argv[4])
end_date = str(sys.argv[5])
time_shift = int(sys.argv[6])


repeated_experiment = False
basin_forcing_dir = Path(data_root / 'forcing_csvs' / f'{basin_id}') # directory where forcing data of all experiments of a specific basin are stored
forcingcsv_dir = Path(data_root / 'forcing_csvs' / f'{basin_id}' / f'{basin_id}_{exp_name}') # directory where forcing data of current experiment is stored
if not os.path.isdir(basin_forcing_dir): # New basin
    os.makedirs(basin_forcing_dir)
    os.makedirs(forcingcsv_dir)
else:
    if os.path.isdir(forcingcsv_dir):
        repeated_experiment = True
        print("Repeated experiment, use the forcing calculated previously to save time...")
    else:
        os.makedirs(forcingcsv_dir)


# Check if the selected basin-id is in Grip-GL basin list
is_train = is_train_basin(basin_id)


# Global variables will be used in the following functions
all_ids = []
all_cells = {}
all_weights = {}

# read the subbasins into the dicts
with open(data_root / f'grid_weights_{basin_id}_subs.txt') as file:
    lines = [line.rstrip() for line in file][7:-1]
for line in lines:
    temp = line.split()
    temp_id = basin_id + '_' + temp[0]
    if temp_id in all_cells:
        all_cells[temp_id].append(int(temp[1]))
        all_weights[temp_id].append(float(temp[2]))
    else:
        all_cells[temp_id] = []
        all_weights[temp_id] = []
        all_cells[temp_id].append(int(temp[1]))
        all_weights[temp_id].append(float(temp[2]))
        all_ids.append(temp_id)

# read the lumped non-gripGL basin into the dicts
with open(data_root / f'grid_weights_{basin_id}.txt') as file:
    lines = [line.rstrip() for line in file][7:-1]

all_ids.append(basin_id)
all_cells[basin_id] = []
all_weights[basin_id] = []

for line in lines:
    temp = line.split()
    all_cells[basin_id].append(int(temp[1]))
    all_weights[basin_id].append(float(temp[2]))


# Extract the target subbasin id from the above 3 global variables
def cells_weights_collector(basin_id : str, cells_dict : dict, weights_dict : dict):
    basin_cells_list = cells_dict[basin_id]
    basin_weights_list = weights_dict[basin_id]
    return tuple(basin_cells_list), tuple(basin_weights_list)


@functools.lru_cache(maxsize = None)
def load_forcings_to_csv(basin_id: str, nc_filename: str, cells: tuple, weights: tuple, merged_file: bool):
    hourly_nc = xr.open_dataset(data_root / 'rdrs_downloads' / nc_filename)

    if merged_file:
        hourly_nc = hourly_nc.sel(time=slice(start_date, end_date))

    one_dim = len(hourly_nc['rlat']) * len(hourly_nc['rlon'])
    variables = list(hourly_nc.keys())
    if 'rotated_pole' in variables:
        variables.remove('rotated_pole')
    temp_ts = hourly_nc['time'].to_series().reset_index(drop=True)
    temp_df = pd.DataFrame(temp_ts)
    temp_df.rename(columns={'time': 'Datetime'}, inplace=True)
    for v in variables:
        if hourly_nc[v].shape != (len(hourly_nc['time']), len(hourly_nc['rlat']), len(hourly_nc['rlon'])):
            continue
        else:
            flat_variable_array = hourly_nc[v].to_numpy().reshape(len(hourly_nc['time']), one_dim) # all values of a given variable
            subset = np.take(flat_variable_array, cells, 1) # select variable values that overlaid with the target subbasin (during the entire study period)
            for timestep in range(len(temp_ts)):
                all_cells_values = subset[timestep]
                variable_value = 0
                for cell_index in range(len(cells)):
                    increment = all_cells_values[cell_index] * weights[cell_index]
                    if np.isnan(increment):
                        variable_value = variable_value + 0
                    else:
                        variable_value = variable_value + increment
                temp_df.loc[timestep, v] = variable_value
    basin_csv = basin_id + '.csv'
    csv_path = Path(forcingcsv_dir / basin_csv)
    if csv_path.is_file():     
        basin_df = pd.read_csv(csv_path, index_col=[0])
        basin_df = pd.concat([basin_df, temp_df]).reset_index(drop=True)
        basin_df.to_csv(csv_path)
    else:
        temp_df.to_csv(csv_path)


def clear():
    load_forcings_to_csv.cache_clear()


def wrapper(basin_id: str):
    basin_cells, basin_weights = cells_weights_collector(basin_id, all_cells, all_weights)

    # subset the rdrs data based on the selected date period
    all_rdrs_files = sorted(os.listdir(data_root / 'rdrs_downloads'))
    
    # check if the netcdfs are raw CaSPAr downaloads or merged
    if len(all_rdrs_files) == 1:
        load_forcings_to_csv(basin_id, all_rdrs_files[0], basin_cells, basin_weights, merged_file = True)
        clear()
    else:
        for idx, filename in enumerate(all_rdrs_files, 0):
            if filename.startswith(start_date.replace('-', '')):
                start_date_index = idx
            if filename.startswith(end_date.replace('-', '')):
                end_date_index = idx
        rdrs_files = all_rdrs_files[start_date_index: end_date_index+1]

        for nc_file in rdrs_files:
            load_forcings_to_csv(basin_id, nc_file, basin_cells, basin_weights, merged_file = False)
            clear()


if not repeated_experiment:
    print("Calculating dynamic forcings for each subbasin, be patient...")
    with mp.Pool(processes = 16) as p:
        max_ = len(all_ids)
        with tqdm(total = max_) as pbar:
            for _ in p.imap_unordered(wrapper, all_ids):
                pbar.update()


    # time shifting the forcing data CSVs
    lf = os.listdir(forcingcsv_dir)
    for csv in tqdm(lf):
        temp_df = pd.read_csv(forcingcsv_dir / csv, index_col=1)
        temp_df.drop(temp_df.columns[[0]], axis = 1, inplace=True)

        temp_df['Datetime'] = pd.to_datetime(temp_df.index).shift(time_shift, freq = 'H')
        temp_df = temp_df.set_index('Datetime')

        temp_df.to_csv(forcingcsv_dir / csv)

    print("Dynamic forcings calculation completed!")


# Move the generated grid weight textfiles to the result folder
os.rename(Path(data_root / f'grid_weights_{basin_id}_subs.txt'), Path(project_root / f'results/{basin_id}/{basin_id}_{exp_name}/tmp_files'/ f'grid_weights_{basin_id}_subs.txt'))
os.rename(Path(data_root / f'grid_weights_{basin_id}.txt'), Path(project_root / f'results/{basin_id}/{basin_id}_{exp_name}/tmp_files'/ f'grid_weights_{basin_id}.txt'))

