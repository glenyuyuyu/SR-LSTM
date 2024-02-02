from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import fiona
from fiona import transform
from osgeo import gdal
import rasterio as rio
from rasterio import mask
from tqdm import *
import os, sys, shutil


class StaticAttributes:

    def __init__(self, args, rootdir):
        self.args = args
        self.rootdir = rootdir

    def run(self):
        basin_id = self.args.watershed
        exp_name = self.args.experiment
        project_root = self.rootdir

        print(f"Prepare to calculate static attributes for the watershed {basin_id}")

        # Read the processed shapefiles
        shapes = {}

        gl_shapefile = gpd.read_file(project_root / f'data/tmp_processed_shapefiles' / f'{basin_id}_subbasins.shp')
        shapefile =  fiona.open(project_root / f'data/tmp_processed_shapefiles' / f'{basin_id}_subbasins.shp')
        shapefile_crs = shapefile.crs

        count = 0

        for i in gl_shapefile.index:
            subbasin_id = gl_shapefile.loc[i, 'Gauge_ID'] + '_' +  str(int(gl_shapefile.loc[i, 'SubId']))
            shapes[subbasin_id] = shapefile[i]['geometry']
            count += 1
        print(f'found {count} subbasins')

        # Add the gauged lumped basin
        gl_shapefile = gpd.read_file(project_root / f'data/tmp_processed_shapefiles'/ f'{basin_id}.shp')
        shapefile =  fiona.open(project_root / f'data/tmp_processed_shapefiles' / f'{basin_id}.shp')
        shapes[basin_id] = shapefile[0]['geometry']
        print(f'found {basin_id} shapefile')


        ####################################
        ### Land Cover 
        ####################################
        print("Estimating Land Cover fractions...")
        lc_classes = {
            1: 'Temperate-or-sub-polar-needleleaf-forest',
            2: 'Sub-polar-taiga-needleleaf-forest',
            3: 'Tropical-or-sub-tropical-broadleaf-evergreen-forest',
            4: 'Tropical-or-sub-tropical-broadleaf-deciduous-forest',
            5: 'Temperate-or-sub-polar-broadleaf-deciduous-forest',
            6: 'Mixed-Forest',
            7: 'Tropical-or-sub-tropical-shrubland',
            8: 'Temperate-or-sub-polar-shrubland',
            9: 'Tropical-or-sub-tropical-grassland',
            10: 'Temperate-or-sub-polar-grassland',
            11: 'Sub-polar-or-polar-shrubland-lichen-moss',
            12: 'Sub-polar-or-polar-grassland-lichen-moss',
            13: 'Sub-polar-or-polar-barren-lichen-moss',
            14: 'Wetland',
            15: 'Cropland',
            16: 'Barren-Lands',
            17: 'Urban-and-Built-up',
            18: 'Water',
            19: 'Snow-and-Ice',
        }
        lc_fractions = pd.DataFrame(columns=lc_classes.values(), index=shapes.keys())

        with rio.open(project_root / 'data' / 'gridded' / 'landcover' / 'NA_NALCMS_2010_v2_land_cover_30m/NA_NALCMS_2010_v2_land_cover_30m.tif') as gridded_ds:
            for basin, shape in tqdm(shapes.items()):
                
                # transform shape to gridded_ds crs
                transformed_shape = transform.transform_geom(shapefile_crs, gridded_ds.crs.data, shape)
                
                # crop to basin outline
                cropped_ds, _ = mask.mask(gridded_ds, [transformed_shape], crop=True,
                                        filled=True, nodata=gridded_ds.nodata)
                cropped_ds = cropped_ds.astype(np.float)
                cropped_ds[cropped_ds == gridded_ds.nodata] = np.nan
                
                for lc_id, lc_class in lc_classes.items():
                    lc_fractions.loc[basin, lc_class] = (cropped_ds == lc_id).sum() / (~np.isnan(cropped_ds)).sum()


        ####################################
        ### Soil
        ####################################
        print("Calculating soil data...")
        soil_sets = ['BD', 'CLAY', 'GRAV', 'OC', 'SAND', 'SILT']
        soil_data = pd.DataFrame(columns=soil_sets, index=shapes.keys())

        for basin, shape in tqdm(shapes.items()):
            # transform shape to gridded_ds crs
            transformed_shape = transform.transform_geom(shapefile_crs, {'init': 'epsg:4326'}, shape)

            for soil_set in soil_sets:

                # soil data is split in two nc files, one containing the first 4 soil layers the other the second 4.
                cropped = []
                for i in [1, 2]:
                    with rio.open(project_root / 'data' / 'gridded' / 'soil' / f'{soil_set}{i}.nc') as gridded_ds:
                            # crop to basin outline
                            cropped_ds, _ = mask.mask(gridded_ds, [transformed_shape], crop=True,
                                                    filled=True, nodata=gridded_ds.nodata)
                            cropped_ds = cropped_ds.astype(np.float)
                            cropped_ds[cropped_ds == gridded_ds.nodata] = np.nan
                            cropped.append(cropped_ds)

                cropped = np.concatenate(cropped, axis=0)
                soil_data.loc[basin, soil_set] = np.nanmean(cropped_ds)

        # To replace all nan values in the soil dataframe
        for basin in soil_data.index.to_list():
            for column in soil_data.columns:
                if np.isnan(soil_data.at[basin, column]):
                    print("found nan value in soil data table, replace it...")
                    column_mode = soil_data.mode()[column][0]
                    soil_data.loc[basin, column] = column_mode


        ####################################
        ### DEM 
        ####################################
        dem_info = pd.DataFrame(columns=['mean_elev', 'mean_slope', 'std_elev', 'std_slope', 'area_km2'],
                                index=shapes.keys())

        with rio.open(project_root / 'data' / 'gridded' / 'dem' / 'na_ca_dem_3s.tif') as gridded_ds:
            for basin, shape in tqdm(shapes.items()):

                # transform shape to gridded_ds crs
                transformed_shape = transform.transform_geom(shapefile_crs, gridded_ds.crs.data, shape)

                # crop to basin outline
                cropped_ds, _ = mask.mask(gridded_ds, [transformed_shape], crop=True,
                                        filled=True, nodata=gridded_ds.nodata)
                cropped_ds = cropped_ds.astype(np.float)
                cropped_ds[cropped_ds == gridded_ds.nodata] = np.nan

                dem_info.loc[basin, 'mean_elev'] = np.nanmean(cropped_ds)
                dem_info.loc[basin, 'std_elev'] = np.nanstd(cropped_ds)

        # leave slope calculation to GDAL. Horizontal units are degrees, vertical units are meters, hence the scale factor (see https://gdal.org/programs/gdaldem.html)
        print("Calculating slope using GDAL...")

        #os.system('/home/glen/anaconda3/envs/srlstm/bin/gdaldem slope -s 111120 ./data/gridded/dem/na_ca_dem_3s.tif ./tmp-slope.tif')
        os.system('~/anaconda3/envs/srlstm/bin/gdaldem slope -s 111120 ./data/gridded/dem/na_ca_dem_3s.tif ./tmp-slope.tif')

        print("Calculating mean slope for each subbasin...")
        with rio.open(project_root / 'tmp-slope.tif', driver='GTiff') as gridded_ds:
            for basin, shape in tqdm(shapes.items()):

                # transform shape to gridded_ds crs
                transformed_shape = transform.transform_geom(shapefile_crs, gridded_ds.crs.data, shape)

                # crop to basin outline
                cropped_ds, _ = mask.mask(gridded_ds, [transformed_shape], crop=True,
                                        filled=True, nodata=gridded_ds.nodata)
                cropped_ds = cropped_ds.astype(np.float)
                cropped_ds[cropped_ds == gridded_ds.nodata] = np.nan

                dem_info.loc[basin, 'mean_slope'] = np.nanmean(cropped_ds)
                dem_info.loc[basin, 'std_slope'] = np.nanstd(cropped_ds)

        os.system('rm ./tmp-slope.tif')


        ####################################
        ### Static Attribute Table
        ####################################
        gl_shapefile = gpd.read_file(project_root / f'data/tmp_processed_shapefiles' / f'{basin_id}_subbasins.shp')

        for i in gl_shapefile.index:
            subbasin_id = gl_shapefile.loc[i, 'Gauge_ID'] + '_' +  str(int(gl_shapefile.loc[i, 'SubId']))
            subbasin_df = gl_shapefile.loc[[i]]
            subbasin_df = subbasin_df.to_crs('ESRI:102017')
            dem_info.loc[subbasin_id, 'area_km2'] = subbasin_df.loc[i, 'geometry'].area * 1e-6
            dem_info.loc[subbasin_id, 'gauge_lat'] = subbasin_df.loc[i, 'centroid_y']
            dem_info.loc[subbasin_id, 'gauge_lon'] = subbasin_df.loc[i, 'centroid_x']

        basin_df = gpd.read_file(project_root / f'data/tmp_processed_shapefiles' / f'{basin_id}.shp')
        basin_df = basin_df.to_crs('ESRI:102017')
        dem_info.loc[basin_id, 'area_km2'] = basin_df.loc[0, 'geometry'].area * 1e-6
        dem_info.loc[basin_id, 'gauge_lat'] = basin_df.loc[0, 'gauge_lat']
        dem_info.loc[basin_id, 'gauge_lon'] = basin_df.loc[0, 'gauge_lon']

        # Join the tables of LC, soil and DEM
        static_attrs = lc_fractions.join(dem_info).join(soil_data)
        static_attrs.index.set_names('basin', inplace=True)
        static_attrs.to_csv(project_root / 'model' / 'attributes' / f'{basin_id}_static_attributes.csv')
        print("Created csv containing static attributes")

        # Move the processed shapefiles to results folder
        shutil.move(str(project_root) + '/data/tmp_processed_shapefiles',\
            str(project_root) + f'/results/{basin_id}/{basin_id}_{exp_name}/tmp_files',\
                copy_function = shutil.copytree)