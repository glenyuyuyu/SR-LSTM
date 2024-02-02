from pathlib import Path
import geopandas as gpd
import os, sys
import warnings
warnings.filterwarnings("ignore")

class SubBasin:
    def __init__(self, args, rootdir):
        self.args = args
        self.rootdir = rootdir

    def run(self):
        basin_id = self.args.watershed
        exp_name = self.args.experiment
        project_root = self.rootdir

        # Check if the result folder is created, if not exit the program
        if not os.path.isdir(project_root / 'results' / f'{basin_id}' / f'{basin_id}_{exp_name}'):
            print("Error:   The result folder does not exist, please make sure everything is setup correctly in the basinmaker script")
            sys.exit(1)

        # Read the rvh (subbasin/HRU info file) to get the list of subbasins
        if not os.path.isfile(project_root / f'data/{basin_id}_{exp_name}/{basin_id}.rvh'):
            os.rename(project_root / f'data/{basin_id}_{exp_name}/model_name.rvh', project_root / f'data/{basin_id}_{exp_name}/{basin_id}.rvh')
        with open(project_root / f'data/{basin_id}_{exp_name}/{basin_id}.rvh', 'r') as rvh:
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
            subid_list.append(int(temp_subid))


        # Create a temporary folder storing the processed shapefiles
        os.makedirs(Path(project_root / f'data/tmp_processed_shapefiles'))

        print("Processing Basinmaker outputs...")
        
        gauge_shp_exist = False # check if the gauge shapefile was produced by BasinMaker
        for file in os.listdir(project_root / f'data/{basin_id}_{exp_name}'):
            if 'catchment_without_merging_lakes' in file:
                subbasins_shp_path = Path(project_root / f'data/{basin_id}_{exp_name}' / file)
            if 'obs_gauges' in file:
                gauge_shp_exist = True
                gauge_shp_path = Path(project_root / f'data/{basin_id}_{exp_name}' / file)
        subbasins_shp = gpd.read_file(subbasins_shp_path)
        subbasins_shp = subbasins_shp.to_crs("EPSG:4326") # set the crs just in case if using OLRP shapefiles
        ls_subids = subbasins_shp['SubId'].to_list()

        # Check if there are duplicate subbasin ids
        if len(ls_subids) != len(set(ls_subids)):
            subbasins_shp = subbasins_shp.dissolve(by='SubId', aggfunc='mean').reset_index()

        subbasins_shp = subbasins_shp[['SubId', 'centroid_x', 'centroid_y', 'geometry']]
        subbasins_shp['Gauge_ID'] = basin_id

        # Define the lumped basin shapefile before removing non-routed subbasin
        basin_shp = subbasins_shp.dissolve(by='Gauge_ID').reset_index()[['Gauge_ID','geometry']]

        # Remove subbasins that are not in the RVH routing network
        subbasins_shp = subbasins_shp[subbasins_shp['SubId'].isin(subid_list)].reset_index(drop=True)

        subbasins_shp.to_file(project_root / f'data/tmp_processed_shapefiles' / f'{basin_id}_subbasins.shp')

        # Obtain the merged basin shapefile without subbasin boundaries
        if gauge_shp_exist:
            gauge_shp = gpd.read_file(gauge_shp_path)
            gauge_shp = gauge_shp.to_crs("EPSG:4326") # set the crs just in case if using OLRP shapefiles
            gauge_shp = gauge_shp.loc[gauge_shp['Obs_NM'] == basin_id].reset_index(drop=True)
            basin_shp['gauge_lat'] = gauge_shp['geometry'].y[0]
            basin_shp['gauge_lon'] = gauge_shp['geometry'].x[0]
            basin_shp['SubId'] = 0
        else:
            basin_shp['gauge_lat'] = float(self.args.gauge_lat)
            basin_shp['gauge_lon'] = float(self.args.gauge_lon)
            basin_shp['SubId'] = 0
        
        basin_shp.to_file(project_root / f'data/tmp_processed_shapefiles' / f'{basin_id}.shp')
            
        print("Shapefiles processed")


        ### Grid Weight Generator
        print("Calculating grid weight...")

        forcing_nc_filename = os.listdir(project_root / 'data/rdrs_downloads')[0]

        os.system(f"python ./scripts/derive_grid_weights.py\
                -i ./data/rdrs_downloads/{forcing_nc_filename}\
                -d 'rlon,rlat' -v 'lon,lat' \
                -r ./data/tmp_processed_shapefiles/{basin_id}_subbasins.shp\
                -o ./data/grid_weights_{basin_id}_subs.txt -a -c 'SubId'")

        
        os.system(f"python ./scripts/derive_grid_weights.py\
                -i ./data/rdrs_downloads/{forcing_nc_filename}\
                -d 'rlon,rlat' -v 'lon,lat' \
                -r ./data/tmp_processed_shapefiles/{basin_id}.shp\
                -o ./data/grid_weights_{basin_id}.txt -a -c 'SubId'")




