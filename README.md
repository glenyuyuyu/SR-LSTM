## Spatially-Recursive LSTM Demo
This example of SR-LSTM only works on the following modules:
1. [NeuralHydrology](https://neuralhydrology.github.io/) Python library in version 1.3.0, 
2. [Raven](http://raven.uwaterloo.ca/) Hydrologic Modelling Framework in version 3.7,
3. RDRS forcing data in version 2.1 downloaded from [CaSPAr](https://caspar-data.ca/),
4. Streamflow/discharge observations that in same format as the CSV files downloaded from [WSC](https://wateroffice.ec.gc.ca/search/historical_e.html)

## Repository Structure
- `basinmaker` -- Folder that contains the DEM and HRU data used for BasinMaker routing delineation
    - `hyd_na_dem_15s.tif` -- DEM data ([HydroSHEDS DEM 15s](https://data.hydrosheds.org/file/hydrosheds-v1-dem/hyd_na_dem_15s.zip))
- `data` 
    - `discharge_obs/` -- Folder that contains the streamflow/discharge observation CSV files
    - `USGS_discharge_data.ipynb` -- Jupyter notebook to convert the format of USGS data to WSC format
    - `forcing_csvs/` -- Folder that will contain the lumped forcings for each subbasins, as well as the lumped forcings for the basin
    - `gridded/` -- Folder that contains the gridded source dataset for calculating the static attributes
        - `dem/` -- DEM data ([HydroSHEDS DEM 3s](https://www.dropbox.com/sh/hmpwobbz9qixxpe/AAAyFvMjPf92oRrw-I-ydyova/HydroSHEDS_DEM/DEM_3s_BIL), merged via `rio merge *.bil na_ca_dem_3s.tif`)
        - `landcover/` -- Landcover data ([NALCMS](http://www.cec.org/north-american-environmental-atlas/land-cover-30m-2020/))
        - `soil/` -- Soil data ([GSDE](http://globalchange.bnu.edu.cn/research/soilw) netCDF files (BD, CLAY, GRAV, OC, SAND, SILT))
    - `rdrs_downloads/` -- Folder that contains the gridded RDRS forcing data
- `model` 
    - `time_series/` -- Folder that will contain the netCDF files with forcings and discharge
    - `attributes/` -- Folder that will contain the CSV files with static attributes
    - `basins/` -- Folder that will contain basin-IDs txtfile
    - `trained_model` -- Folder that contains the trained LSTM lumped model files
- `raven` -- Folder that contains the Raven files
- `results` -- Folder that will contain the simulation results
- `scripts` -- Folder that contains the SR-LSTM codes
    - `derive_grid_weights.py` -- Script of [Grid-Weights-Generator](https://github.com/julemai/GridWeightsGenerator)
    - `ensemble2netcdf.py` -- Script that takes the pickled ensemble reults file and creates one netCDF submission file with predictions
    
## Setup Steps
1. Create a virtual environment in Anaconda3 using the provided `srlstm_environment.yml`
2. Download the RDRS v2.1 forcings from CaSPAr and place them in `data/rdrs_downloads/`
3. Download the gridded static attribute data and place them in corresponding folder under `data/gridded/` and `basinmaker/`
4. Place streamflow observation CSVs in `data/discharge_obs/`
5. Place validated NeuralHydrology LSTM model files in `model/trained_model/`
6. Create a txtfile named as`train_basin.txt` which contains the IDs of basins used to train the LSTM model, and place it in `model/basins/`
7. Edit the input parameters in `scripts/run.py`, and run the script to start the simulation

## Input parameters with example
- `--watershed` -- the gauged basin ID
- `--experiment` -- define the delineation scheme
    - 'default' -- default mode, the threshold for minimum drainage area of subbasins will be 10% of the total basin area, lakes smaller than 5km^2 will be removed
    - 'lumped' -- lumped mode, there will be no discretization
    - 'allsublake' -- all-in mode, all subbasins and lakes in the routing product will be preserved
    - 'MDAx_LAy' -- change x and y to define the threshold for subbasin area and lake area. e.g., MDA200_LA10 uses 200km^2 as the threshold to merge subbasins and 10km^2 to remove lakes 
- `gauge_lat` -- latitude of the basin gauge under the WGS 84 geospatial reference system. Note that Google Maps uses this geospatial reference system.
- `gauge_lon` -- longitude of the basin gauge under the WGS 84 geospatial reference system.
- `start_date` `end_date` -- the time range for simulation, use the format yyyy-mm-dd
- `save_forcing` -- optional, used if you want to save the generated subbasin-level forcings

Example:
    os.system(f"python ./scripts/main.py  --watershed 02KB001\
                                        --experiment MDA200_LA5\
                                        --gauge_lat 45.886111\
                                        --gauge_lon -77.315278\
                                        --start_date 1980-01-01\
                                        --end_date 1981-12-31\
                                        --save_forcing")