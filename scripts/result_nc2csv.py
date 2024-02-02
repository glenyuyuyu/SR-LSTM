import xarray as xr
import pandas as pd
from pathlib import Path
import os, sys
from tqdm import *
from utils import *
import warnings
warnings.filterwarnings("ignore")


class Result2CSV:

    def __init__(self, args, rootdir):
        self.args = args
        self.rootdir = rootdir


    def export_basin_to_csv(self, ID : str, XRDS : xr.Dataset):
        input_start_date = self.args.start_date
        start_date = str(int(input_start_date.split('-')[0]) + 1) + '-' + input_start_date.split('-')[1] + '-' + input_start_date.split('-')[2]
        end_date = self.args.end_date

        sub_ds = XRDS.sel(basin=ID)
        time_var = sub_ds.variables['datetime']
        
        q_obs = sub_ds.variables['qobs_m3_per_s_obs']
        q_obs_ts = pd.Series(q_obs, index=time_var)
        Q_list = [q_obs_ts]

        q_sim = sub_ds.variables['qobs_m3_per_s_sim']
        q_sim_ts = pd.Series(q_sim, index=time_var)
        Q_list.append(q_sim_ts)

        combined = pd.concat(Q_list, axis=1)
        combined.columns = ['Observed', 'Predicted']
        combined = combined.loc[f'{start_date}':f'{end_date}']
        combined.to_csv(self.rootdir / 'model' / 'predictions' / f'{ID}.csv', index=True, header=True)
    

    def run(self):
        basin_id = self.args.watershed
        project_root = self.rootdir

        # Move the ensembled result to prediction folder
        os.rename(Path(project_root / 'model' /  'trained_model' / 'test_ensemble_results.nc'), Path(project_root / 'model' / 'predictions' / f'{basin_id}_ensemble_results.nc'))
        os.rename(Path(project_root / 'model' / 'trained_model' / 'test_ensemble_results.p'), Path(project_root / 'model' / 'predictions' / f'{basin_id}_ensemble_results.p'))
        print("LSTM forward prediction completed!")

        result_path = Path(project_root / 'model' /'predictions' / f'{basin_id}_ensemble_results.nc')
        result = xr.open_dataset(result_path)
        ids = result.variables['basin'].to_numpy()

        print("Converting prediction results to csv for each subbasin...")
        for id in tqdm(ids):
            self.export_basin_to_csv(id, result)


        ''' Archive used files '''
        # Move used static attribute csv files to the prediction folder
        os.replace(Path(project_root / 'model/attributes' / f'{basin_id}_climate_indices.csv'), Path(project_root / 'model/predictions' / 'files' / f'{basin_id}_climate_indices.csv'))
        os.replace(Path(project_root / 'model/attributes' / f'{basin_id}_static_attributes.csv'), Path(project_root / 'model/predictions' / 'files' / f'{basin_id}_static_attributes.csv'))

        # Move used time series netcdf files to the prediction folder
        ncs = os.listdir(project_root / 'model/time_series')
        for nc in ncs:
            os.replace(Path(project_root / 'model/time_series' / nc), Path(project_root / 'model/predictions' / 'files' / nc))

        # Move the subbasin list txtfile to the prediction folder
        os.replace(Path(project_root / 'model/basins' / f'{basin_id}_subbasins.txt'), Path(project_root / 'model/predictions' / 'files' / f'{basin_id}_subbasins.txt'))