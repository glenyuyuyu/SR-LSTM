from pathlib import Path
import numpy as np
import pandas as pd
import os, sys
import shutil
from utils import *
import warnings
warnings.filterwarnings("ignore")

class RoutingAnalysis:

    def __init__(self, args, rootdir):
        self.args = args
        self.rootdir = rootdir
    
    def run(self):
        basin_id = self.args.watershed
        exp_name = self.args.experiment
        save_forcing = self.args.save_forcing
        project_root = self.rootdir

        lstm_root = Path(project_root / 'model')
        raven_dir = Path(project_root / f'raven/{basin_id}_{exp_name}')

        # Find the gauged subbasin
        gauged = find_gauged_subbasin(basin_id, exp_name)

        # load the Raven routing results
        raven_hydrograph = pd.read_csv(raven_dir/ 'output' / f'{basin_id}_Hydrographs.csv', index_col='date')

        # find the column of Raven-prediction
        for col_index in range(len(raven_hydrograph.columns)):
                col_name = raven_hydrograph.columns[col_index]
                if col_name == 'sub' + gauged + ' [m3/s]':
                    pred_col = col_index

        raven_hydrograph.rename(columns = {raven_hydrograph.columns[pred_col] : 'Raven_predicted'}, inplace = True) # By default Raven produce the prediction at outlet subbasin on the second last column in hydrograph.csv
        raven_hydrograph.drop(raven_hydrograph.columns.difference(['Raven_predicted']), 1, inplace=True)

        # resample the simulation results from hourly to daily
        raven_hydrograph['Raven_predicted'] = raven_hydrograph['Raven_predicted'].shift(-1)
        raven_hydrograph['date'] = pd.to_datetime(raven_hydrograph.index)
        raven_hydrograph = raven_hydrograph.set_index('date')
        raven_hydrograph = raven_hydrograph.resample("1D").mean()

        
        # load the LSTM prediction
        lstm_hydrograph = pd.read_csv(lstm_root / 'predictions' / f'{basin_id}.csv', index_col=[0])

        lstm_hydrograph.index.name = 'date'
        lstm_hydrograph.rename(columns = {'Predicted' : 'LSTM_predicted'}, inplace=True)
        lstm_hydrograph['date'] = pd.to_datetime(lstm_hydrograph.index)
        lstm_hydrograph = lstm_hydrograph.set_index('date')

        basin_df = lstm_hydrograph.join(raven_hydrograph)

        
        basin_df.dropna(axis=0, inplace=True)
        obs = basin_df['Observed'].to_numpy()

        lstm_sim = basin_df['LSTM_predicted'].to_numpy()
        lstm_kge = kling_gupta_efficiency(lstm_sim, obs)

        raven_sim = basin_df['Raven_predicted'].to_numpy()
        raven_kge = kling_gupta_efficiency(raven_sim, obs)

        results = {}
        results[f'{self.args.start_date} to {self.args.end_date}'] = [lstm_kge, raven_kge]


        # Record the experiment result
        result_dir = Path(project_root / f'results/{basin_id}/{basin_id}_{exp_name}')

        # Move raven files to the result folder
        raven_files = os.listdir(raven_dir)
        for file in raven_files:
            shutil.move(raven_dir / file, result_dir / 'Raven_files')
        os.rmdir(raven_dir)

        # Move LSTM files to the result folder
        LSTM_files = os.listdir(lstm_root / 'predictions')
        for file in LSTM_files:
            shutil.move(lstm_root / 'predictions' / file, result_dir / 'LSTM_files')
        os.rmdir(lstm_root / 'predictions')

        results_df = pd.DataFrame(data=results, index=['Pretrained LSTM', 'Raven Routing']).T
        results_df.to_csv(Path(result_dir / f'{basin_id}_KGE.csv'),index=True, header=True)
        basin_df.to_csv(Path(result_dir / f'{basin_id}_hydrographs.csv'),index=True, header=True)

        # Move the basinmaker-generated routing files into result
        source = Path(project_root / 'data' / f'{basin_id}_{exp_name}')
        destination = Path(project_root / 'results' / basin_id / f'{basin_id}_{exp_name}' / 'tmp_files' / 'bm_routing')
        shutil.move(source, destination)

        # Delete the calcualted subbasin forcings
        if not save_forcing:
            shutil.rmtree(project_root / 'data' / 'forcing_csvs' / basin_id / f'{basin_id}_{exp_name}')

        print(f"Experiment {basin_id}_{exp_name} completed!!!!!!!!!!!!")