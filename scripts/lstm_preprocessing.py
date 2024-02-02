from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import *
import xarray as xr
import os, shutil
import yaml
import warnings
from utils import *
warnings.filterwarnings("ignore")

from neuralhydrology.datautils.climateindices import calculate_dyn_climate_indices
from neuralhydrology.datautils.pet import get_priestley_taylor_pet

class Forcing2TimeSeries:
    def __init__(self, args, rootdir):
        self.args = args
        self.rootdir = rootdir
        self.static_attributes = pd.read_csv(rootdir / 'model' /  'attributes' / f'{self.args.watershed}_static_attributes.csv', index_col=[0])


    def load_discharge(self, basin: str, lumped: str) -> pd.Series:
        if basin != lumped:
            return None
        
        df = pd.read_csv(self.rootdir / f'data/discharge_obs/{basin}_discharge.csv', index_col=2, header=1)
        discharge_df = df[df['PARAM'] == 1]
        discharge_df.rename(columns={'Value':'Q'}, inplace=True)
        discharge_df.index = pd.to_datetime(discharge_df.index)
        discharge = discharge_df['Q']

        area = self.static_attributes.loc[basin, 'area_km2'] * 1000_000  # km2 to m2
        discharge.loc[discharge < 0] = np.nan
        discharge.index.name = 'date'
        
        return discharge * (60 * 60 * 24) * 1000.0 / area  # m3/s to mm/day

    
    def create_timeseries(self):
        basin_id = self.args.watershed
        exp_name = self.args.experiment
        project_root = self.rootdir

        #static_attributes = pd.read_csv(project_root / 'model' /  'attributes' / f'{basin_id}_static_attributes.csv', index_col=[0])
        static_attributes = self.static_attributes


        ####################################
        ### Generating the time series files for LSTM 
        ####################################
        print("Generating time series files for LSTM prediction...")
        clim_indices = {}

        no_forcing_basins = []
        no_discharge_basins  = []

        for subbasin in tqdm(sorted(static_attributes.index)):
            
            basin_rdrs_file = project_root / 'data/forcing_csvs' / basin_id / f'{basin_id}_{exp_name}' / f'{subbasin}.csv'
            if not basin_rdrs_file.exists():
                no_forcing_basins.append(subbasin)
                continue
                
            # The aggregated RDRS forcings are already in local standard time (UTC-5) and selected date period
            forcings = pd.read_csv(basin_rdrs_file, index_col=0, parse_dates=[0], skipinitialspace=True)
            forcings.index.name = 'date'
            climidx_forcings = forcings

            if forcings.isna().all(axis=None):
                no_forcing_basins.append(subbasin)
                continue

            lat = static_attributes.loc[subbasin, 'gauge_lat']

            # resample to daily values: sum(precip), min/max(temp), mean for all other variables
            for i, forcing_set in enumerate([climidx_forcings, forcings]):
                daily_resampled = forcing_set.resample('1D')
                daily_forcings = daily_resampled.mean()

                # precip
                daily_forcings['RDRS_v2.1_A_PR0_SFC'] = daily_resampled['RDRS_v2.1_A_PR0_SFC'].sum(min_count=1)
                # temp
                daily_forcings['min_RDRS_v2.1_P_TT_1.5m'] = daily_resampled['RDRS_v2.1_P_TT_1.5m'].min()
                daily_forcings['max_RDRS_v2.1_P_TT_1.5m'] = daily_resampled['RDRS_v2.1_P_TT_1.5m'].max()
                daily_forcings['potential_evapotranspiration'] = \
                    get_priestley_taylor_pet(daily_forcings['min_RDRS_v2.1_P_TT_1.5m'].values,
                                            daily_forcings['max_RDRS_v2.1_P_TT_1.5m'].values,
                                            daily_forcings['RDRS_v2.1_P_FB_SFC'].values,  # shortwave radiation
                                            lat=lat,
                                            elev=static_attributes.loc[subbasin, 'mean_elev'],
                                            doy=daily_forcings.index.dayofyear.values,
                                            )

                if i == 0:
                    # since window_length is length of forcings, there will only be one date returned, so we can do .iloc[0]
                    clim_indices[subbasin] = calculate_dyn_climate_indices(daily_forcings['RDRS_v2.1_A_PR0_SFC'] * 1000,  # m to mm
                                                                        daily_forcings['max_RDRS_v2.1_P_TT_1.5m'],
                                                                        daily_forcings['min_RDRS_v2.1_P_TT_1.5m'],
                                                                        daily_forcings['potential_evapotranspiration'],
                                                                        window_length=len(daily_forcings)).iloc[0]
        
            # Add discharge
            discharge = self.load_discharge(subbasin, basin_id)
            daily_forcings['qobs_mm_per_day'] = discharge if discharge is not None else np.nan
            if discharge is None:
                no_discharge_basins.append(subbasin)

            xr_forcings_and_discharge = xr.Dataset.from_dataframe(daily_forcings)
            xr_forcings_and_discharge.to_netcdf(project_root / 'model' /  'time_series' / f'{subbasin}.nc')

        print(f'No forcings for basins:  ({len(no_forcing_basins)}) {no_forcing_basins}')
        print(f'No discharge for basins: ({len(no_discharge_basins)}) {no_discharge_basins}')


        # Generate the climate indices as additional static attributes
        print("Calculating climate indices...")
        indices = pd.DataFrame(clim_indices).T
        indices.index.set_names('basin', inplace=True)
        indices.columns = [col.split('_dyn')[0] for col in indices.columns]
        indices.to_csv(project_root / 'model' /'attributes' / f'{basin_id}_climate_indices.csv')


        # Generate the basin list textfile
        ids = []
        for file in os.listdir(project_root / 'model' / 'time_series'):
            id = file[:-3]
            ids.append(id)
        ids.sort()
        with open(project_root / 'model' / 'basins' / f'{basin_id}_subbasins.txt', 'w') as f:
            for id in ids:
                f.write(f"{id}\n")

        # Create a folder to store the LSTM files/predictions for the testing basin
        os.makedirs(project_root / 'model' / 'predictions' / 'files')
        print('Ready for running LSTM prediction')
    
    
    def config_training(self):
        basin_id = self.args.watershed
        project_root = self.rootdir
        lstm_root = Path(project_root / 'model')

        config_files = Path(project_root / 'model' /  'trained_model').glob('*/*.yml')

        print("Updating the pretrained model configuration files...")
        for file in config_files:
            model_name = str(file).split('/')[-2]
            
            with open(file) as inf:
                configs = yaml.load(inf, Loader=yaml.SafeLoader)
            
            configs['commit_hash'] = None
            #configs['data_dir'] = '/home/glen/works/sr_lstm/model'
            configs['data_dir'] = str(lstm_root)

            #configs['img_log_dir'] = f'/home/glen/works/sr_lstm/model/trained_model/{model_name}/img_log'
            configs['img_log_dir'] = str(lstm_root) + f'/trained_model/{model_name}/img_log'

            #configs['run_dir'] = f'/home/glen/works/sr_lstm/model/trained_model/{model_name}'
            configs['run_dir'] = str(lstm_root) + f'/trained_model/{model_name}'
            
            
            configs['test_basin_file'] = f'basins/{basin_id}_subbasins.txt'

            #configs['train_dir'] = f'/home/glen/works/sr_lstm/model/trained_model/{model_name}/train_data'
            configs['train_dir'] = str(lstm_root) + f'/trained_model/{model_name}/train_data'
            
            with open(file, 'w') as outf:
                yaml.dump(configs, outf, sort_keys=False, default_flow_style=False)


        # Delete the testing files from previous experiment if there is any
        trained_dir = Path(project_root / 'model/trained_model')
        for root, subdirs, files in os.walk(trained_dir):
            for folder in subdirs:
                if folder == 'test':
                    shutil.rmtree(os.path.join(root, folder))
                    print("Found previous testing folder, removing...")


        # Running the LSTM forward prediction
        print("Start to make predictions...")