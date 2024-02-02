from tqdm import tqdm
import pandas as pd
import geopandas as gpd
from pathlib import Path
import os, sys
import shutil
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class LSTM2Raven:
    
    def __init__(self, args, rootdir):
        self.args = args
        self.rootdir = rootdir
    
    def run(self):
        basin_id = self.args.watershed
        exp_name = self.args.experiment

        input_start_date = self.args.start_date
        start_date = str(int(input_start_date.split('-')[0]) + 1) + '-' + input_start_date.split('-')[1] + '-' + input_start_date.split('-')[2]
        
        end_date = self.args.end_date
        project_root = self.rootdir

        data_root = Path(project_root / 'data')
        lstm_root = Path(project_root / 'model')
        raven_root = Path(project_root / 'raven')


        # Move the template files to the model folder
        if not os.path.isdir(raven_root / f'{basin_id}_{exp_name}'):
            shutil.copytree(Path(raven_root / 'model_template'), Path(raven_root / f'{basin_id}_{exp_name}'))
            os.rename(Path(raven_root / f'{basin_id}_{exp_name}'/ 'model_name.rvc'), Path(raven_root / f'{basin_id}_{exp_name}'/ f'{basin_id}.rvc'))
            os.rename(Path(raven_root / f'{basin_id}_{exp_name}'/ 'model_name.rvi'), Path(raven_root / f'{basin_id}_{exp_name}'/ f'{basin_id}.rvi'))
            os.rename(Path(raven_root / f'{basin_id}_{exp_name}'/ 'model_name.rvp'), Path(raven_root / f'{basin_id}_{exp_name}'/ f'{basin_id}.rvp'))

            with open(raven_root / f'{basin_id}_{exp_name}'/ f'{basin_id}.rvi', 'r') as f:
                lines = f.readlines()
            
            lines[3] = f':StartDate             {start_date} 00:00:00\n'
            lines[4] = f':EndDate               {end_date} 00:00:00\n'
            lines[7] = f':RunName               {basin_id}\n'
            lines[22] = f':EvaluationPeriod TESTING {start_date} {end_date}'

            with open(raven_root / f'{basin_id}_{exp_name}'/ f'{basin_id}.rvi', 'w') as f:
                f.writelines(lines)



        # Move the required routing info from its Basinmaker outputs, including ({basin_id}.rvh, channel_properties.rvp, Lakes.rvh)
        routing_dir = Path(data_root / f'{basin_id}_{exp_name}')
        for file in [f'{basin_id}.rvh', 'channel_properties.rvp', 'Lakes.rvh']:
            shutil.copy(Path(routing_dir / file), Path(raven_root / f'{basin_id}_{exp_name}' / file))


        # Use LSTM-predictions to calculate the global parameter :AvgAnnualRunoff and write to the rvp file
        lstm_preds = pd.read_csv(lstm_root / 'predictions' / f'{basin_id}.csv', index_col=0)
        lstm_preds.index = pd.to_datetime(lstm_preds.index)
        lstm_preds = lstm_preds.loc[f'{start_date}':f'{end_date}']
        AvgAnnualRunoff_m3_s = lstm_preds['Predicted'].resample('Y').sum().mean(skipna=True)
        static_attributes = pd.read_csv(lstm_root / 'predictions/files' / f'{basin_id}_static_attributes.csv', index_col=[0])
        area = static_attributes.loc[f'{basin_id}', 'area_km2'] * 1000_000  # km2 to m2
        AvgAnnualRunoff = int(AvgAnnualRunoff_m3_s * (60 * 60 * 24) * 1000.0 / area)  # m3/s to mm/day


        with open(raven_root / f'{basin_id}_{exp_name}'/ f'{basin_id}.rvp', 'r') as f:
            lines = f.readlines()
        lines[0] = lines[0].replace("???", str(AvgAnnualRunoff))

        with open(raven_root / f'{basin_id}_{exp_name}'/ f'{basin_id}.rvp', 'w') as f:
            f.writelines(lines)

        # Read the rvh (subbasin/HRU info file) to get the list of subbasins
        with open(raven_root / f'{basin_id}_{exp_name}'/ f'{basin_id}.rvh', 'r') as rvh:
            alllines = rvh.readlines()

        for i in range(len(alllines)):
            if ":SubBasins" in alllines[i]:
                start_index = i+3
            if ":EndSubBasins" in alllines[i]:
                end_index = i
                break

        subid_list = []
        for li in range(start_index, end_index):
            temp_subid = alllines[li].split()[0]
            subid_list.append(temp_subid)

        # Make sure the most downstream subbasin is set as 'gauged'
        for li in range(start_index, end_index):
            downstream_id = alllines[li].split()[2]
            if downstream_id == '-1':
                if alllines[li].split()[5] == '0':
                    alllines[li] = alllines[li][:-2] + '1\n'
                    with open(raven_root / f'{basin_id}_{exp_name}'/ f'{basin_id}.rvh', 'w') as rvh:
                        rvh.writelines(alllines)

        # find the gauged subbasin
        gauged_sub = subid_list[-1] # By default Basinmaker records the gauged subbasin lastly, but make sure to double check it in the rvh file
        

        # Read the observed discharge data and write to rvt file
        basin_df = pd.read_csv(lstm_root / 'predictions' / f'{basin_id}.csv', index_col=0)
        observed = basin_df['Observed'].to_list()
        

        # Create the observation data rvt file
        with open(raven_root / f'{basin_id}_{exp_name}' / f'{basin_id}_Qobs.rvt', 'w') as f:
            f.write(f':ObservationData  HYDROGRAPH  {gauged_sub}  m3/s\n')
            f.write(f'{start_date}  00:00:00  1  {len(observed)}\n')
            for date in observed:
                if np.isnan(date): # if the observed value is missing, flag it with -1.2345 as Raven requires
                    f.write('  -1.2345\n')
                else:
                    f.write(f'  {date}\n')
            f.write(':EndObservationData\n')


        # Write the link to the main Raven model RVT file
        with open(raven_root / f'{basin_id}_{exp_name}' / f'{basin_id}.rvt', 'w') as f:
            f.write('# Observed discharge data\n')
            f.write(f':RedirectToFile   {basin_id}_Qobs.rvt\n')


        # Use the gauged subbasin id to create a dummy temperature rvt file just for Raven to run properly
        static_attributes = pd.read_csv(lstm_root / 'predictions' / 'files' / f'{basin_id}_static_attributes.csv', index_col=[0])
        lat = static_attributes.loc[f'{basin_id}_{gauged_sub}', 'gauge_lat']
        lon = static_attributes.loc[f'{basin_id}_{gauged_sub}', 'gauge_lon']
        elev = static_attributes.loc[f'{basin_id}_{gauged_sub}', 'mean_elev']

        with open(raven_root / f'{basin_id}_{exp_name}' / 'dummy_temp.rvt', 'w') as f:
            f.write(f':Data  TEMP_AVE  C\n')
            f.write(f'{start_date}  00:00:00  1  {len(observed)}\n')
            i = 0
            while i < len(observed):
                f.write('  0\n')
                i += 1
            f.write(':EndData\n')

        # Link the dummy temperature info to the main Raven model rvt file
        with open(raven_root / f'{basin_id}_{exp_name}' / f'{basin_id}.rvt', 'a') as f:
            f.write('\n')
            f.write('# Dummy temperature data\n')
            f.write(f':Gauge {basin_id}\n')
            f.write(f'  :Latitude   {lat}\n')
            f.write(f'  :Longitude  {lon}\n')
            f.write(f'  :Elevation  {elev}\n')
            f.write(f':RedirectToFile   dummy_temp.rvt\n')
            f.write(':EndGauge\n')


        # Use the subbasin list to read the corresponding csv that contains LSTM prediction results
        print("loading subbasin prediction result to Raven model...")
        csv_list = os.listdir(lstm_root / 'predictions')
        for subid in tqdm(subid_list):
            for csv in csv_list:
                if basin_id + '_'  + subid in csv:
                    temp_df = pd.read_csv(lstm_root / 'predictions' / csv, index_col=0)
                    predicted = temp_df['Predicted'].to_list()
                    
                    # write to a rvt file as (forcing - precipitation)
                    static_attributes = pd.read_csv(lstm_root / 'predictions' / 'files' / f'{basin_id}_static_attributes.csv', index_col=[0])
                    
                    lat = static_attributes.loc[f'{basin_id}_{subid}', 'gauge_lat']
                    lon = static_attributes.loc[f'{basin_id}_{subid}', 'gauge_lon']
                    elev = static_attributes.loc[f'{basin_id}_{subid}', 'mean_elev']
                    
                    # Add the subbasin info to the main Raven model rvt file
                    with open(raven_root / f'{basin_id}_{exp_name}' / f'{basin_id}.rvt', 'a') as f:
                        f.write('\n')
                        f.write('# Subbasin streamflow as precipitation forcing\n')
                        f.write(f':Gauge {basin_id}\n')
                        f.write(f'  :Latitude   {lat}\n')
                        f.write(f'  :Longitude  {lon}\n')
                        f.write(f'  :Elevation  {elev}\n')
                        f.write(f'  :RedirectToFile input/sub{subid}.rvt  \n')
                        f.write(':EndGauge\n')
                    
                    area = static_attributes.loc[f'{basin_id}_{subid}', 'area_km2'] * 1000_000  # km2 to m2
                    predicted = [i * (60 * 60 * 24) * 1000.0 / area for i in predicted] # m3/s to mm/day

                    with open(raven_root / f'{basin_id}_{exp_name}' / 'input' / f'sub{subid}.rvt', 'w') as f:
                        f.write(f':Data  PRECIP  mm/d\n')
                        f.write(f'{start_date}  00:00:00  1  {len(predicted)}\n')
                        for date in predicted:
                            if np.isnan(date): # if the LSTM does not properly predict on a subbasin, which results in all nan values, change it to all zeros (as forcing input to Raven) 
                                f.write(f'  0\n')
                            else:
                                f.write(f'  {date}\n')
                        f.write(':EndData\n')


        # Run the Raven routing-only mode
        os.system(f'./raven/Raven.exe ./raven/{basin_id}_{exp_name}/{basin_id} -o ./raven/{basin_id}_{exp_name}/output')

