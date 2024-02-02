from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import functools
import multiprocessing as mp
import os, sys
from tqdm import *
import geopandas as gpd
import warnings
warnings.filterwarnings("ignore")


project_root = Path(os.getcwd())
output_dir = Path(project_root / 'forcing_csvs')


with open(project_root / 'gauge_ids.txt') as file:
    basin_ids = [line.rstrip() for line in file]


def grid_weight_process(basin_id : str):
    # Read the combined shapefile containing polygons of all basins, change as needed
    all_basins_shp = gpd.read_file(project_root / f'shapefiles' / 'GRIP_GL_141_calibration_catchment_info.shp')
    sel = all_basins_shp.loc[all_basins_shp['Gauge_ID'] == basin_id]
    if len(sel.index) == 0:
        all_basins_shp = gpd.read_file(project_root / f'shapefiles' / 'GRIP_GL_71_validation_catchment_info.shp')
        sel = all_basins_shp.loc[all_basins_shp['Gauge_ID'] == basin_id]
    sel = sel.reset_index(drop=True)
    basins_shp = sel[['SubId', 'INSIDE_Y','INSIDE_X','Gauge_ID', 'geometry']]
    #basins_shp = basins_shp.dissolve(by='SubId', aggfunc='mean').reset_index()
    basins_shp = basins_shp.dissolve(by='SubId').reset_index()
    basins_shp['Gauge_ID'] = basin_id
    basins_shp.rename(columns={'INSIDE_X':'centroid_x', 'INSIDE_Y':'centroid_y'}, inplace=True)
    basin_shp = basins_shp.dissolve(by='Gauge_ID')
    basin_shp.to_file(project_root / 'shapefiles' / 'basins' / f'{basin_id}.shp')
    os.system(f"python derive_grid_weights.py\
                        -i rdrs_downloads/1980010112.nc\
                        -d 'rlon,rlat' -v 'lon,lat' \
                        -r shapefiles/basins/{basin_id}.shp\
                        -o shapefiles/basins/{basin_id}_grid_weights.txt -a -c 'SubId'")


def cells_weights_collector(basin_id : str):
    cells = []; weights = []
    with open(project_root / 'shapefiles' / 'basins' / f'{basin_id}_grid_weights.txt') as file:
        lines = [line.rstrip() for line in file][7:-1]
    for line in lines:
        temp = line.split()
        cells.append(int(temp[1]))
        weights.append(float(temp[2]))
    return tuple(cells), tuple(weights)


@functools.lru_cache(maxsize = None)
def load_one_day_to_csv(basin_id: str, nc_filename: str, cells: tuple, weights: tuple):
    daily_nc = xr.open_dataset(project_root / 'rdrs_downloads' / nc_filename)
    one_dim = len(daily_nc['rlat']) * len(daily_nc['rlon'])
    variables = list(daily_nc.keys())
    variables.remove('rotated_pole')
    temp_ts = daily_nc['time'].to_series().reset_index(drop=True)
    temp_df = pd.DataFrame(temp_ts)
    temp_df.rename(columns={'time': 'Datetime'}, inplace=True)
    for v in variables:
        flat_variable_array = daily_nc[v].to_numpy().reshape(len(daily_nc['time']), one_dim) # all values of a given variable
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
    csv_dir = Path(output_dir / f'{basin_id}')
    if not csv_dir.is_dir():
        csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(csv_dir / basin_csv)
    if csv_path.is_file():     
        basin_df = pd.read_csv(csv_path, index_col=[0])
        basin_df = pd.concat([basin_df, temp_df]).reset_index(drop=True)
        basin_df.to_csv(csv_path)
    else:
        temp_df.to_csv(csv_path)


def clear():
    load_one_day_to_csv.cache_clear()


def wrapper(basin_id: str):
    grid_weight_process(basin_id)
    basin_cells, basin_weights = cells_weights_collector(basin_id)
    for nc_file in sorted(os.listdir(project_root / 'rdrs_downloads')):
        load_one_day_to_csv(basin_id, nc_file, basin_cells, basin_weights)
        clear()

n_p = len(basin_ids)
n_p = 36

with mp.Pool(processes = n_p) as p:
    max_ = len(basin_ids)
    with tqdm(total = max_) as pbar:
        for _ in p.imap_unordered(wrapper, basin_ids):
            pbar.update()

'''         
with mp.Pool(processes = n_p) as p:
    p.map(wrapper, basin_ids)
'''