from pathlib import Path
import shutil
from basinmaker import basinmaker 
import pandas as pd
import fiona
from fiona import transform
import rasterio as rio
from rasterio import mask
import os, sys, warnings
import wget
import geopandas as gpd
import time
from branca.colormap import linear
import argparse
from utils import *

from basinmaker.postprocessing.downloadpd import Download_Routing_Product_For_One_Gauge
from basinmaker.postprocessing.downloadpdptspurepy import Download_Routing_Product_From_Points_Or_LatLon
warnings.filterwarnings('ignore')


class RoutingNetwork:
    def __init__(self, args, rootdir):
        self.args = args
        self.rootdir = rootdir

    
    def run(self):
        basin_id = self.args.watershed
        exp_name = self.args.experiment
        project_root = self.rootdir
        basinmaker_root = Path(project_root / 'basinmaker')

        routing_provided = self.args.routing_provided
        if routing_provided:
            assert os.path.isdir(project_root / 'data/routing_networks' / f"{basin_id}_routing"), "no routing network found"
            version_number = 'v1-0'
            
        is_train = is_train_basin(basin_id)
        if is_train:
            print("WARNING: The basin was used to train the lumped LSTM.")

        is_default = False
        if 'default' in exp_name:
            is_default = True
            min_subbasin_drainage_area = None
            lake_filter_threshold = 5
            print("The basin will be modelled in default mode")

        elif 'lumped' in exp_name:
            min_subbasin_drainage_area = 999999999
            lake_filter_threshold = 999999999
            print("The basin will be modelled in lumped mode")

        elif 'allsublake' in exp_name:
            min_subbasin_drainage_area = 0
            lake_filter_threshold = 0
            print("The basin will be modelled in Keep-all-subbasins-lakes mode")

        else: # Defined by MDA***_LA***
            min_subbasin_drainage_area = int(exp_name.split('_')[0][3:]) # unit in km2
            print(f"Minimum subbasin draignage area is {min_subbasin_drainage_area} km^2")
            if 'nolake' in exp_name:
                lake_filter_threshold = 999999999
                print("The basin will be delineated without lakes")
            elif 'alllake' in exp_name:
                lake_filter_threshold = 0
                print("The basin will be delineated with all lakes")
            else:
                lake_filter_threshold = int(exp_name.split('_')[1][2:]) # unit in km2
                print(f"Lake filtering threshold is {lake_filter_threshold} km^2")


        if not routing_provided:
            '''Download a routing product for River catchment'''
            ##################################################################
            # First option using gauge name to download the routing product region/sub-region that contains the watershed defined by the gauge or the lat-long co-ordinates. 
            ##################################################################

            # Define the product name 
                # 'OLRP' to use the Ontario Lake-River Routing Product(version 1.0)
                # 'NALRP' to use the North American Lake-River Routing Product (version 2.1)

            product_name = 'NALRP'

            if product_name == 'NALRP':
                version_number = 'v2-1'
            elif product_name == 'OLRP':
                version_number = 'v1-0'

            # Define the gauge name
            subid, product_path = Download_Routing_Product_For_One_Gauge(gauge_name = basin_id, product_name = product_name)

            if subid == '-1' or product_path == '#':
                print("The gauge is not included in the routing product, using gauge coordinates...")
                ##################################################################
                # Second option using lat lon to download the routing product.
                # The Lat,Lon should under the WGS 84 geospatial reference system. Note that Google Maps uses this geospatial reference system. 
                # (EPSG:4326)
                ##################################################################

                # Use the input gauge coordinates
                Lat = [float(self.args.gauge_lat)]
                Lon = [float(self.args.gauge_lon)]
                subid, product_path = Download_Routing_Product_From_Points_Or_LatLon(product_name = product_name,Lat = Lat,Lon = Lon)

            '''Extract drainage areas of gauge'''
            print('Extract drainage areas of gauge...')
            # BasinMaker needs the ID of subbasin (subId) which the gauge is situated in.
            subid_of_interested_gauges = [subid]

            # Define another folder that will save the outputs 
            extraction_folder = basinmaker_root / basin_id / 'extraction'

            # Initialize the basinmaker
            bm = basinmaker.postprocess()

            # extract subregion of the routing product
            bm.Select_Subregion_Of_Routing_Structure(
                path_output_folder = extraction_folder,
                routing_product_folder = product_path,
                most_down_stream_subbasin_ids=subid_of_interested_gauges,
                most_up_stream_subbasin_ids=[-1],               # -1: extract to the most-upstream (headwater) subbasin; other subbasin ID: extract the areas from the outlet to the provided subbasin.
                gis_platform="purepy",
            )

            # Delete the downloaded routing product and zip file
            shutil.rmtree(product_path)
            os.remove(product_path + '.zip')
        
        # if the routing network is provided
        else:
            extraction_folder = basinmaker_root / basin_id / 'extraction'
            shutil.copytree(project_root / 'data/routing_networks' / f"{basin_id}_routing", extraction_folder)
            bm = basinmaker.postprocess()


        # If using default delineation method, set the threshold for minimum drainage area of subbasins
        if is_default:
            print(f'Default delineation method: 10% of the total basin area as the threshold of minimum drainage area of subbasins')
            extraction = Path(extraction_folder)
            subbasins_shp = gpd.read_file(extraction / f'catchment_without_merging_lakes_{version_number}.shp')
            subbasins_shp['Gauge_ID'] = basin_id
            basin_shp = subbasins_shp.dissolve(by='Gauge_ID').reset_index()[['Gauge_ID','geometry']].to_crs("EPSG:6933")
            min_subbasin_drainage_area = int((basin_shp['geometry'].area / 1000000).values[0] / 10) # use 10% of the total basin area as the subbasin drainage area threshold
            print(f"The minimum drainage area of subbasins is {min_subbasin_drainage_area} km^2")


        '''Simplify the routing product by filtering lakes'''
        print('Simplify the routing product by filtering lakes...')
        # define another folder that will save the outputs the gauge_name is defined in previous section. 
        filter_lake_folder = Path(basinmaker_root / basin_id / 'filter_lakes')


        # define a list containing HyLakeId ID of lakes of interest that are not be removed even if their area is smaller than lake area threshold
        # remove small lakes

        interested_lake_ids = [] # Change this to keep certain lakes
       
        bm.Remove_Small_Lakes(
            path_output_folder = filter_lake_folder,
            routing_product_folder = extraction_folder,
            connected_lake_area_thresthold= lake_filter_threshold,                 # unit km2, this is to remove lakes with area < 5km2 (= 5km2 will not be removed)
            non_connected_lake_area_thresthold= lake_filter_threshold,             # unit km2, this is to remove lakes with area < 5km2 (= 5km2 will not be removed)
            selected_lake_ids=interested_lake_ids,
            gis_platform="purepy",
        )    


        '''Simplify the routing product by increasing size of subbasins'''
        print('Simplify the routing product by increasing size of subbasins...')
        # define another folder that will save the outputs
        drainage_area_folder = Path(basinmaker_root / basin_id / 'drainage_area')
        

        # Initialize the basinmaker 
        bm = basinmaker.postprocess()

        # remove river reaches and increase size of subbasin
        bm.Decrease_River_Network_Resolution(
            path_output_folder = drainage_area_folder,
            routing_product_folder = filter_lake_folder,  # if not filtering any lakes in Section 4, use outing_product_folder = extraction_folder
            minimum_subbasin_drainage_area = min_subbasin_drainage_area,           # unit in km2. Definition of this threshold is subbasins (as well as river reaches) with drainage areas < this value, say 50 km2, will be removed. For a subbasin, it will be merged to another subbasin to get a larger subbasin that meets the drainage area threshold.
            gis_platform="purepy",
        )


        '''Crop the DEM according to basin outline'''
        print("Cropping DEM for the basin outline...")

        # Read the subbasin shapefile and merge it
        basin_shp = gpd.read_file(basinmaker_root / basin_id/ 'drainage_area' / f'catchment_without_merging_lakes_{version_number}.shp')
        basin_shp = basin_shp.to_crs("EPSG:4326")
        basin_shp['Gauge_ID'] = basin_id
        basin_shp = basin_shp.dissolve(by='Gauge_ID').reset_index()[['Gauge_ID','geometry']]
        basin_shp.to_file(basinmaker_root / basin_id /'tmp_basin_outline.shp')

        # Get the bounding box of the basin outline
        shapefile = fiona.open(basinmaker_root / basin_id / 'tmp_basin_outline.shp')
        shapefile_crs = shapefile.crs
        shape = shapefile[0]['geometry']
        bbox = shapefile.bounds

        # Read the DEM of North America
        NA_DEM = rio.open(project_root / 'data/gridded/dem' / 'na_ca_dem_3s.tif')
        transformed_shape = fiona.transform.transform_geom(shapefile_crs, NA_DEM.crs.data, shape)
        cropped_ds, _ = mask.mask(NA_DEM, [transformed_shape], crop=True, filled=True, nodata=0)
        cropped_ds = cropped_ds.reshape(cropped_ds.shape[1], cropped_ds.shape[2])
        transform = rio.transform.from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], cropped_ds.shape[1], cropped_ds.shape[0])

        # Save the new DEM raster
        with rio.open(
            basinmaker_root / basin_id / f'{basin_id}_dem.tif',
            'w',
            driver='GTiff',
            height=cropped_ds.shape[0],
            width=cropped_ds.shape[1],
            count=1,
            dtype=cropped_ds.dtype,
            crs=NA_DEM.crs,
            transform=transform,
        ) as dst:
            dst.write(cropped_ds, 1)


        '''Create HRUs using multiple geo-spatial data layers'''
        print('Create HRUs using multiple geo-spatial data layers...')

        landuse_info = Path(basinmaker_root / 'HRU_data' / 'landcover_info.csv')
        soil_info = Path(basinmaker_root / 'HRU_data' / 'soil_info.csv')
        veg_info = Path(basinmaker_root / 'HRU_data' / 'veg_info.csv')

        ### Create HRUs ###

        # define another folder that will save the outputs 
        # gauge_name is a variable defined in section 3. Redefine here if necessary.
        HRU_output_folder = Path(basinmaker_root / basin_id / 'HRU_simplified')

        ################################################################################
        # Please note that when thresholds for lake sizes to retain are so large that either 
        # all connected lake polygons are removed (e.g. sl_connected_lake_***.shp does not exist) 
        # and/or all non-connected lake polygons are removed (e.g. sl_non_connected_lake_***.shp 
        # does not exist), we need to specify this by using '#' below.  
        # path_connect_lake_polygon            = '#'
        # path_non_connect_lake_polygon        = '#'
        # So please check if these .shp files exist! 
        ################################################################################
        bm = basinmaker.postprocess()

        # Check if there is any lakes
        connect_lake = '#'
        non_connect_lake = '#'
        if os.path.isfile(os.path.join(drainage_area_folder, "sl_connected_lake_" + version_number + ".shp")):
            connect_lake = os.path.join(drainage_area_folder, "sl_connected_lake_" + version_number + ".shp")

        if os.path.isfile(os.path.join(drainage_area_folder, "sl_non_connected_lake_" + version_number + ".shp")):
            non_connect_lake = os.path.join(drainage_area_folder, "sl_non_connected_lake_" + version_number + ".shp")


        bm.Generate_HRUs(
            path_output_folder=str(HRU_output_folder),
            path_subbasin_polygon         = os.path.join(str(drainage_area_folder), "finalcat_info_" + version_number + ".shp"), 
            path_connect_lake_polygon     = connect_lake,     # change to '#' when the connected lake polygon does not exist
            path_non_connect_lake_polygon = non_connect_lake, # change to '#' when the non connected lake polygon does not exist 
            path_landuse_polygon='#',
            path_soil_polygon   ='#',
            path_other_polygon_1='#',
            path_landuse_info=landuse_info,
            path_soil_info   =soil_info,
            path_veg_info    =veg_info,
            path_to_dem= str(basinmaker_root) + f'/{basin_id}' + f'/{basin_id}_dem.tif',
            area_ratio_thresholds = [0.1, 0.2, 0.2],     # In frist trail, use [0,0,0] to get the default HRU map. In second trial, try using [0.1, 0.2, 0.3] -ish (larger than 0) to get simplified HRU maps.
            gis_platform="purepy",
            projected_epsg_code = 'EPSG:3161',  # EPSG:3161 corresponds to the projected coordinate system NAD83 / Ontario MNR Lambert. Used for aspect/area calculation.
            pixel_size = 30         # User-defined grid size in m. We recommend using 30 m for OLRRP and 90 m for NA. The unit follows the coordinate system of the routing network polygons.
        )


        '''Produce Raven-required inputs'''
        print('Produce Raven-required inputs...')
        # define another folder that will save the outputs 
        raven_model_dir = Path(basinmaker_root / basin_id / 'Raven_inputs')

        bm = basinmaker.postprocess()

        bm.Generate_Raven_Model_Inputs(
            path_hru_polygon         = os.path.join(HRU_output_folder, "finalcat_hru_info.shp"),
            model_name            = basin_id,                             # This is used for naming the output files.
            subbasingroup_names_channel   =["Allsubbasins"],                # A subbasin group will be created in the rvh file for simultaneous manipulation in Raven modeling.
            subbasingroup_length_channel   =[-1],
            subbasingroup_name_lake      =["AllLakesubbasins"],
            subbasingroup_area_lake      =[-1],
            path_output_folder         = raven_model_dir,
            aspect_from_gis          = 'purepy',
        )

        raven_model_dir = raven_model_dir / 'RavenInput'


        # Delete the automatically-generated empty rvp file
        shutil.rmtree(str(raven_model_dir) + '/obs')
        os.remove(os.path.join(raven_model_dir , f'{basin_id}.rvp'))


        '''Create the SR model required input files'''
        print('Create the SR model required input files...')
        sr_input_folder = Path(os.getcwd() + f'/{basin_id}_{exp_name}')
        os.makedirs(sr_input_folder )

        basin_files = os.listdir(drainage_area_folder)
        raven_files = os.listdir(raven_model_dir)

        for file in basin_files:
            if os.path.isfile(drainage_area_folder / file):
                shutil.copy(Path(drainage_area_folder / file), Path(sr_input_folder  / file))

        for file in raven_files:
            if os.path.isfile(raven_model_dir / file):
                shutil.copy(Path(raven_model_dir / file), Path(sr_input_folder / file))

        # check the rvh file make sure the gauged subbasin is marked as 1 not 0
        with open(sr_input_folder  / f'{basin_id}.rvh', 'r') as f:
            alllines = f.readlines()

        for i in range(len(alllines)):
            if ":SubBasins" in alllines[i]:
                start_index = i+3
            if ":EndSubBasins" in alllines[i]:
                end_index = i
                break

        for li in range(start_index, end_index):
            line_list = alllines[li].split()
            downstream_id = line_list[2]
            if downstream_id == '-1':
                gauged_indicator = line_list[5]
                if gauged_indicator == '0':
                    old_line = alllines[li]
                    alllines[li] = old_line[:-2] + '1' + old_line[-2+1:]
                    with open(sr_input_folder  / f'{basin_id}.rvh', 'w') as f:
                        f.writelines(alllines)


        shutil.rmtree(basinmaker_root / basin_id)

        # Check if the basin folder is created in data root with discharge data
        if not os.path.isfile(project_root / f'data/discharge_obs/{basin_id}_discharge.csv'):
            print("No discharge data found, program stops")
            shutil.rmtree(sr_input_folder)
            sys.exit(1)


        # Initialize the folder for storing the results
        if not os.path.isdir(project_root / 'results' / f'{basin_id}' / f'{basin_id}_{exp_name}'):
            os.makedirs(project_root / 'results' / f'{basin_id}' / f'{basin_id}_{exp_name}')
        else:
            print("WARNING: Repeated experiment! The previous experiment files and results will be overwritted")
            shutil.rmtree(project_root / 'results' / f'{basin_id}' / f'{basin_id}_{exp_name}')
            os.makedirs(project_root / 'results' / f'{basin_id}' / f'{basin_id}_{exp_name}')

        os.makedirs(project_root / 'results' / f'{basin_id}' / f'{basin_id}_{exp_name}' / 'LSTM_files')
        os.makedirs(project_root / 'results' / f'{basin_id}' / f'{basin_id}_{exp_name}' / 'Raven_files')
        os.makedirs(project_root / 'results' / f'{basin_id}' / f'{basin_id}_{exp_name}' / 'tmp_files')

        # Move the input folder
        shutil.move(str(sr_input_folder), project_root / 'data', copy_function = shutil.copytree)
        