#!/usr/bin/env python
from __future__ import print_function

# Copyright 2016-2020 Juliane Mai - juliane.mai(at)uwaterloo.ca
#
# License
# This file is part of Juliane Mai's personal code library.
#
# Juliane Mai's personal code library is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Juliane Mai's personal code library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with Juliane Mai's personal code library.  If not, see <http://www.gnu.org/licenses/>.
#

# TESTING
#
# --------------------------------------------------
# GRIP-GL version with streamflow gauge ID given (attribute "Obs_NM" in shapefile) --> -b 02LE024
# --------------------------------------------------

# ------------------------
#        VIC
# ------------------------
#    run derive_grid_weights.py -i example/input_VIC/VIC_streaminputs.nc -d "lon,lat" -v "lon,lat" -r example/maps/HRUs_coarse.shp -b 02LE024 -o example/input_VIC/GridWeights.txt

# ------------------------
#        MESH
# ------------------------
#    run derive_grid_weights.py -i example/input_MESH/RFF_H_GRD.nc      -d "rlon,rlat" -v "longitude,latitude" -r example/maps/HRUs_coarse.shp -b 02LE024 -o example/input_MESH/GridWeights_RFF_H_GRD.txt
#    run derive_grid_weights.py -i example/input_MESH/DRAINSOL_H_GRD.nc -d "rlon,rlat" -v "longitude,latitude" -r example/maps/HRUs_coarse.shp -b 02LE024 -o example/input_MESH/GridWeights_DRAINSOL_H_GRD.txt

# ------------------------
#        SWAT
# ------------------------
#    run derive_grid_weights.py -i example/input_SWAT/OTT_sub.shp -f "NetCDF_col" -r example/maps/HRUs_coarse.shp -b 02LE024 -o example/input_SWAT/GridWeights_OTT_sub.txt
#    run derive_grid_weights.py -i example/input_SWAT/OTT_sub.shp -f "NetCDF_col" -r example/maps/HRUs_coarse.shp -b 02LE024 -o example/input_SWAT/GridWeights_OTT_sub.txt -e 0.42

# ------------------------
#        1D coordinates
# ------------------------
#    run derive_grid_weights.py -i example/input_ERA5/era5-crop.nc -d "longitude,latitude" -v "longitude,latitude" -r example/maps/HRUs_coarse.shp -b 02LE024 -o example/input_ERA5/GridWeights_ERA5.txt


# --------------------------------------------------
# GRIP-GL version with subbasin ID given (attribute "SubId" in shapefile)  --> -s 7202
# --------------------------------------------------

# ------------------------
#        VIC
# ------------------------
#    run derive_grid_weights.py -i example/input_VIC/VIC_streaminputs.nc -d "lon,lat" -v "lon,lat" -r example/maps/HRUs_coarse.shp -s 7202 -o example/input_VIC/GridWeights.txt

# ------------------------
#        MESH
# ------------------------
#    run derive_grid_weights.py -i example/input_MESH/RFF_H_GRD.nc      -d "rlon,rlat" -v "longitude,latitude" -r example/maps/HRUs_coarse.shp -s 7202 -o example/input_MESH/GridWeights_RFF_H_GRD.txt
#    run derive_grid_weights.py -i example/input_MESH/DRAINSOL_H_GRD.nc -d "rlon,rlat" -v "longitude,latitude" -r example/maps/HRUs_coarse.shp -s 7202 -o example/input_MESH/GridWeights_DRAINSOL_H_GRD.txt

# ------------------------
#        SWAT
# ------------------------
#    run derive_grid_weights.py -i example/input_SWAT/OTT_sub.shp -f "NetCDF_col" -r example/maps/HRUs_coarse.shp -s 7202 -o example/input_SWAT/GridWeights_OTT_sub.txt
#    run derive_grid_weights.py -i example/input_SWAT/OTT_sub.shp -f "NetCDF_col" -r example/maps/HRUs_coarse.shp -s 7202 -o example/input_SWAT/GridWeights_OTT_sub.txt -e 0.42

# ------------------------
#        1D coordinates
# ------------------------
#    run derive_grid_weights.py -i example/input_ERA5/era5-crop.nc -d "longitude,latitude" -v "longitude,latitude" -r example/maps/HRUs_coarse.shp -s 7202 -o example/input_ERA5/GridWeights_ERA5.txt


# read netCDF files
import netCDF4 as nc4

# command line arguments
import argparse

# checking file paths and file extensions
from pathlib import Path

# to perform numerics
import numpy as np

# read shapefiles and convert to GeoJSON and WKT
import geopandas as gpd

# get equal-area projection and derive overlay
from   osgeo   import ogr
from   osgeo   import osr
from   osgeo   import __version__ as osgeo_version



input_file           = "example/input_VIC/VIC_streaminputs.nc"
dimname              = "rlon,rlat"
varname              = "lon,lat"
routinginfo          = "example/maps/HRUs_coarse.shp"
basin                = None  # e.g. "02LE024"
SubId                = None  # e.g. 7202
output_file          = "GriddedForcings.txt"
doall                = False
key_colname          = "HRU_ID"
key_colname_model    = None
area_error_threshold = 0.05

parser      = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
              description='''Convert files from ArcGIS raster format into NetDF file usable in CaSPAr.''')
parser.add_argument('-i', '--input_file', action='store',
                    default=input_file, dest='input_file', metavar='input_file',
                    help='Either (A) Example NetCDF file containing at least 1D or 2D latitudes and 1D or 2D longitudes where grid needs to be representative of model outputs that are then required to be routed. Or (B) a shapefile that contains shapes of subbasins and one attribute that is defining its index in the NetCDF model output file (numbering needs to be [0 ... N-1]).')
parser.add_argument('-d', '--dimname', action='store',
                    default=dimname, dest='dimname', metavar='dimname',
                    help='Dimension names of longitude (x) and latitude (y) (in this order). Example: "rlon,rlat", or "x,y"')
parser.add_argument('-v', '--varname', action='store',
                    default=varname, dest='varname', metavar='varname',
                    help='Variable name of 2D longitude and latitude variables in NetCDF (in this order). Example: "lon,lat".')
parser.add_argument('-r', '--routinginfo', action='store',
                    default=routinginfo, dest='routinginfo', metavar='routinginfo',
                    help='Shapefile that contains all information for the catchment of interest (and maybe some more catchments).')
parser.add_argument('-b', '--basin', action='store',
                    default=basin, dest='basin', metavar='basin',
                    help='Basin of interest (corresponds to "Gauge_ID" in shapefile given with -r). Either this or SubId ID (-s) needs to be given. Can be a comma-separated list of basins, e.g., "02LB005,02LB008".')
parser.add_argument('-s', '--SubId', action='store',
                    default=SubId, dest='SubId', metavar='SubId',
                    help='SubId of most downstream subbasin (containing usually a gauge station) (corresponds to "SubId" in shapefile given with -r). Either this or basin ID (-b) needs to be given. Can be a comma-separated list of SubIds, e.g., "7399,7400".')
parser.add_argument('-o', '--output_file', action='store',
                    default=output_file, dest='output_file', metavar='output_file',
                    help='File that will contain grid weights for Raven.')
parser.add_argument('-a', '--doall', action='store_true',
                    default=doall, dest='doall',
                    help='If given, all HRUs found in shapefile are processed. Overwrites settings of "-b" and "-s". Default: not set (False).')
parser.add_argument('-c', '--key_colname', action='store',
                    default=key_colname, dest='key_colname', metavar='key_colname',
                    help='Name of column in shapefile containing unique key for each dataset. This key will be used in output file. This setting is only used if "-a" option is used. "Default: "HRU_ID".')
parser.add_argument('-f', '--key_colname_model', action='store',
                    default=key_colname_model, dest='key_colname_model', metavar='key_colname_model',
                    help='Attribute name in input_file shapefile (option -i) that defines the index of the shape in NetCDF model output file (numbering needs to be [0 ... N-1]). Example: "NetCDF_col".')
parser.add_argument('-e', '--area_error_threshold', action='store',
                    default=area_error_threshold, dest='area_error_threshold', metavar='area_error_threshold',
                    help='Threshold (as fraction) of allowed mismatch in areas between subbasins from shapefile (-r) and overlay with grid-cells or subbasins (-i). If error is smaller than this threshold the weights will be adjusted such that they sum up to exactly 1. Raven will exit gracefully in case weights do not sum up to at least 0.95. Default: 0.05.')

args                 = parser.parse_args()
input_file           = args.input_file
dimname              = np.array(args.dimname.split(','))
varname              = np.array(args.varname.split(','))
routinginfo          = args.routinginfo
basin                = args.basin
SubId                = args.SubId
output_file          = args.output_file
doall                = args.doall
key_colname          = args.key_colname
key_colname_model    = args.key_colname_model
area_error_threshold = float(args.area_error_threshold)

if not(SubId is None):

    SubId = [ np.int(ss.strip()) for ss in SubId.split(',') ]

if (SubId is None) and (basin is None) and not(doall):
    raise ValueError("Either gauge ID (option -b; e.g., 02AB003) or SubId ID (option -s; e.g., 7173) specified in shapefile needs to be given. You specified none. This basin will be the most downstream gridweights of all upstream subbasins will be added automatically.")

if ( not(SubId is None) ) and ( not(basin is None) ) and not(doall):
    raise ValueError("Either gauge ID (option -b; e.g., 02AB003) or SubId ID (option -s; e.g., 7173) specified in shapefile needs to be specified. You specified both. This basin will be the most downstream gridweights of all upstream subbasins will be added automatically.")

if not(doall):
    key_colname = "HRU_ID"    # overwrite any settimg made for this column name in case doall is not set

del parser, args


# better dont chnage that ever
crs_lldeg = 4326        # EPSG id of lat/lon (deg) coordinate referenence system (CRS)
crs_caea  = 3573        # EPSG id of equal-area    coordinate referenence system (CRS)


def create_gridcells_from_centers(lat, lon):

    # create array of edges where (x,y) are always center cells
    nlon = np.shape(lon)[1]
    nlat = np.shape(lat)[0]
    lonh = np.empty((nlat+1,nlon+1), dtype=float)
    lath = np.empty((nlat+1,nlon+1), dtype=float)
    tmp1 = [ [ (lat[ii+1,jj+1]-lat[ii,jj])/2 for jj in range(nlon-1) ] + [ (lat[ii+1,nlon-1]-lat[ii,nlon-2])/2 ] for ii in range(nlat-1) ]
    tmp2 = [ [ (lon[ii+1,jj+1]-lon[ii,jj])/2 for jj in range(nlon-1) ] + [ (lon[ii+1,nlon-1]-lon[ii,nlon-2])/2 ] for ii in range(nlat-1) ]
    dlat = np.array(tmp1 + [ tmp1[-1] ])
    dlon = np.array(tmp2 + [ tmp2[-1] ])
    lonh[0:nlat,0:nlon] = lon - dlon
    lath[0:nlat,0:nlon] = lat - dlat

    # make lat and lon one column and row wider such that all
    lonh[nlat,0:nlon] = lonh[nlat-1,0:nlon] + (lonh[nlat-1,0:nlon] - lonh[nlat-2,0:nlon])
    lath[nlat,0:nlon] = lath[nlat-1,0:nlon] + (lath[nlat-1,0:nlon] - lath[nlat-2,0:nlon])
    lonh[0:nlat,nlon] = lonh[0:nlat,nlon-1] + (lonh[0:nlat,nlon-1] - lonh[0:nlat,nlon-2])
    lath[0:nlat,nlon] = lath[0:nlat,nlon-1] + (lath[0:nlat,nlon-1] - lath[0:nlat,nlon-2])
    lonh[nlat,nlon]   = lonh[nlat-1,nlon-1] + (lonh[nlat-1,nlon-1] - lonh[nlat-2,nlon-2])
    lath[nlat,nlon]   = lath[nlat-1,nlon-1] + (lath[nlat-1,nlon-1] - lath[nlat-2,nlon-2])

    return [lath,lonh]

def shape_to_geometry(shape_from_jsonfile,epsg=None):

    # converts shape read from shapefile to geometry
    # epsg :: integer EPSG code

    ring_shape = ogr.Geometry(ogr.wkbLinearRing)

    for ii in shape_from_jsonfile:
        ring_shape.AddPoint_2D(ii[0],ii[1])
    # close ring
    ring_shape.AddPoint_2D(shape_from_jsonfile[0][0],shape_from_jsonfile[0][1])

    poly_shape = ogr.Geometry(ogr.wkbPolygon)
    poly_shape.AddGeometry(ring_shape)

    if not( epsg is None):
        source = osr.SpatialReference()
        source.ImportFromEPSG(crs_lldeg)       # usual lat/lon projection

        target = osr.SpatialReference()
        target.ImportFromEPSG(epsg)       # any projection to convert to

        transform = osr.CoordinateTransformation(source, target)
        poly_shape.Transform(transform)

    return poly_shape

def check_proximity_of_envelops(gridcell_envelop, shape_envelop):

    # checks if two envelops are in proximity (intersect)

    # minX  --> env[0]
    # maxX  --> env[1]
    # minY  --> env[2]
    # maxY  --> env[3]

    if  ((gridcell_envelop[0] <= shape_envelop[1]) and (gridcell_envelop[1] >= shape_envelop[0]) and
         (gridcell_envelop[2] <= shape_envelop[3]) and (gridcell_envelop[3] >= shape_envelop[2])):

        grid_is_close = True

    else:

        grid_is_close = False

    return grid_is_close

def check_gridcell_in_proximity_of_shape(gridcell_edges, shape_from_jsonfile):

    # checks if a grid cell falls into the bounding box of the shape
    # does not mean it intersects but it is a quick and cheap way to
    # determine cells that might intersect

    # gridcell_edges = [(lon1,lat1),(lon2,lat2),(lon3,lat3),(lon4,lat4)]
    # shape_from_jsonfile

    min_lat_cell  = np.min([ii[1] for ii in gridcell_edges])
    max_lat_cell  = np.max([ii[1] for ii in gridcell_edges])
    min_lon_cell  = np.min([ii[0] for ii in gridcell_edges])
    max_lon_cell  = np.max([ii[0] for ii in gridcell_edges])

    lat_shape = np.array([ icoord[1] for icoord in shape_from_jsonfile ])     # is it lat???
    lon_shape = np.array([ icoord[0] for icoord in shape_from_jsonfile ])     # is it lon???

    min_lat_shape = np.min(lat_shape)
    max_lat_shape = np.max(lat_shape)
    min_lon_shape = np.min(lon_shape)
    max_lon_shape = np.max(lon_shape)

    if  ((min_lat_cell <= max_lat_shape) and (max_lat_cell >= min_lat_shape) and
         (min_lon_cell <= max_lon_shape) and (max_lon_cell >= min_lon_shape)):

        grid_is_close = True

    else:

        grid_is_close = False

    return grid_is_close


def derive_2D_coordinates(lat_1D, lon_1D):

    # nlat = np.shape(lat_1D)[0]
    # nlon = np.shape(lon_1D)[0]

    # lon_2D =              np.array([ lon_1D for ilat in range(nlat) ], dtype=float32)
    # lat_2D = np.transpose(np.array([ lat_1D for ilon in range(nlon) ], dtype=float32))

    lon_2D =              np.tile(lon_1D, (lat_1D.size, 1))
    lat_2D = np.transpose(np.tile(lat_1D, (lon_1D.size, 1)))

    return lat_2D, lon_2D


if ( Path(input_file).suffix == '.nc'):

    # -------------------------------
    # Read NetCDF
    # -------------------------------
    print(' ')
    print('   (1) Reading NetCDF (grid) data ...')

    nc_in = nc4.Dataset(input_file, "r")
    lon      = nc_in.variables[varname[0]][:]
    lon_dims = nc_in.variables[varname[0]].dimensions
    lat      = nc_in.variables[varname[1]][:]
    lat_dims = nc_in.variables[varname[1]].dimensions
    nc_in.close()


    if len(lon_dims) == 1 and len(lat_dims) == 1:

        # in case coordinates are only 1D (regular grid), derive 2D variables

        print('   >>> Generate 2D lat and lon fields. Given ones are 1D.')

        lat, lon = derive_2D_coordinates(lat,lon)
        lon_dims_2D = lat_dims + lon_dims
        lat_dims_2D = lat_dims + lon_dims
        lon_dims = lon_dims_2D
        lat_dims = lat_dims_2D

    elif len(lon_dims) == 2 and len(lat_dims) == 2:

        # Raven numbering is (numbering starts with 0 though):
        #
        #      [      1      2      3   ...     1*nlon
        #        nlon+1 nlon+2 nlon+3   ...     2*nlon
        #           ...    ...    ...   ...     ...
        #           ...    ...    ...   ...  nlat*nlon ]
        #
        # --> Making sure shape of lat/lon fields is like that
        #

        if np.all(np.array(lon_dims) == dimname[::1]):
            lon = np.transpose(lon)
            print('   >>> switched order of dimensions for variable "{0}"'.format(varname[0]))
        elif np.all(np.array(lon_dims) == dimname[::-1]):
            print('   >>> order of dimensions correct for variable "{0}"'.format(varname[0]))
        else:
            print('   >>> Dimensions found {0} does not match the dimension names specified with (-d): {1}'.format(lon_dims,dimname))
            raise ValueError('STOP')

        if np.all(np.array(lat_dims) == dimname[::1]):
            lat = np.transpose(lat)
            print('   >>> switched order of dimensions for variable "{0}"'.format(varname[1]))
        elif np.all(np.array(lat_dims) == dimname[::-1]):
            print('   >>> order of dimensions correct for variable "{0}"'.format(varname[1]))
        else:
            print('   >>> Dimensions found {0} does not match the dimension names specified with (-d): {1}'.format(lat_dims,dimname))
            raise ValueError('STOP')

    else:

        raise ValueError(
            "The coord variables must have the same number of dimensions (either 1 or 2)"
        )

    lath, lonh    = create_gridcells_from_centers(lat, lon)

    nlon       = np.shape(lon)[1]
    nlat       = np.shape(lat)[0]
    nshapes    = nlon * nlat

elif ( Path(input_file).suffix == '.shp'):

    # -------------------------------
    # Read Shapefile
    # -------------------------------
    print(' ')
    print('   (1) Reading Shapefile (grid) data ...')

    model_grid_shp     = gpd.read_file(input_file)
    model_grid_shp     = model_grid_shp.to_crs(epsg=crs_caea)           # WGS 84 / North Pole LAEA Canada

    nshapes    = model_grid_shp.geometry.count()    # number of shapes in model "discretization" shapefile (i.e. model grid-cells; not basin-discretization shapefile)
    nlon       = 1        # only for consistency
    nlat       = nshapes  # only for consistency

else:

    print("File extension found: {}".format(input_file.split('.')[-1]))
    raise ValueError('Input file needs to be either NetCDF (*.nc) or a Shapefile (*.shp).')


# -------------------------------
# Read Basin shapes and all subbasin-shapes (from toolbox)
# -------------------------------
print(' ')
print('   (2) Reading shapefile data ...')

shape     = gpd.read_file(routinginfo)
# shape     = shape.to_crs(epsg=crs_lldeg)        # this is lat/lon in degree
shape     = shape.to_crs(epsg=crs_caea)           # WGS 84 / North Pole LAEA Canada

# check that key column contains only unique values
keys = np.array(list(shape[key_colname]))
# keys_uniq = np.unique(keys)
# if len(keys_uniq) != len(keys):
#     raise ValueError("The attribute of the shapefile set to contain only unique identifiers ('{}') does contain duplicate keys. Please specify another column (option -c '<col_name>') and use the option to process all records contained in the shapefile (-a).".format(key_colname))


# select only relevant basins/sub-basins
if not(doall):

    if not(basin is None):    # if gauge ID is given

        basins     = [ bb.strip() for bb in basin.split(',') ]
        idx_basins = [ list(np.where(shape['Obs_NM']==bb)[0]) for bb in basins ]

        # find corresponding SubId
        SubId = [ np.int(shape.loc[idx_basin].SubId) for idx_basin in idx_basins ]
        print("   >>> found gauge at SubId = ",SubId)

    if not(SubId is None): # if basin ID is given

        old_SubIds = []
        for SI in SubId:

            old_SubId     = []
            new_SubId     = [ SI ]

            while len(new_SubId) > 0:

                old_SubId.append(new_SubId)
                new_SubId = [ list(shape.loc[(np.where(shape['DowSubId']==ii))[0]].SubId) for ii in new_SubId ]  # find all upstream catchments of these new basins
                new_SubId = list(np.unique([item for sublist in new_SubId for item in sublist])) # flatten list and make entries unique

            old_SubId   = np.array([item for sublist in old_SubId for item in sublist],dtype=np.int)  # flatten list
            old_SubIds += list(old_SubId)

        old_SubIds = list( np.sort(np.unique(old_SubIds)) )

        idx_basins = [ list(np.where(shape['SubId']==oo)[0]) for oo in old_SubIds ]
        idx_basins = [ item for sublist in idx_basins for item in sublist ]  # flatten list
        idx_basins = list(np.unique(idx_basins))                             # getting only unique list indexes

else: # all HRUs to be processed

    idx_basins = list(np.arange(0,len(shape)))


# make sure HRUs are only once in this list
hrus = np.array( shape.loc[idx_basins][key_colname] ) #[sort_idx]

idx_basins_unique = []
hrus_unique       = []
for ihru,hru in enumerate(hrus):

    if not( hru in hrus_unique ):

        hrus_unique.append(hrus[ihru])
        idx_basins_unique.append(idx_basins[ihru])

idx_basins = idx_basins_unique
hrus       = hrus_unique


# order according to values in "key_colname"; just to make sure outputs will be sorted in the end
sort_idx = np.argsort(shape.loc[idx_basins][key_colname])
print('   >>> HRU_IDs found = ',list(np.array( shape.loc[idx_basins][key_colname] )[sort_idx]),'  (total: ',len(idx_basins),')')

# reduce the shapefile dataset now to only what we will need
shape     = shape.loc[np.array(idx_basins)[sort_idx]]

# indexes of all lines in df
keys       = shape.index
nsubbasins = len(keys)

# initialize
coord_catch_wkt = {}

# loop over all subbasins and transform coordinates into equal-area projection
for kk in keys:

    ibasin = shape.loc[kk]

    poly                = ibasin.geometry
    try:
        coord_catch_wkt[kk] = ogr.CreateGeometryFromWkt(poly.to_wkt())
    except:
        coord_catch_wkt[kk] = ogr.CreateGeometryFromWkt(poly.wkt)

# -------------------------------
# construct all grid cell polygons
# -------------------------------
if ( Path(input_file).suffix == '.nc'):

    print(' ')
    print('   (3) Generate shapes for NetCDF grid cells ...')

    grid_cell_geom_gpd_wkt = [ [ [] for ilon in range(nlon) ] for ilat in range(nlat) ]
    for ilat in range(nlat):
        if ilat%10 == 0:
            print('   >>> Latitudes done: {0} of {1}'.format(ilat,nlat))

        for ilon in range(nlon):

            # -------------------------
            # EPSG:3035   needs a swap before and after transform ...
            # -------------------------
            # gridcell_edges = [ [lath[ilat,ilon]    , lonh[ilat,  ilon]    ],            # for some reason need to switch lat/lon that transform works
            #                    [lath[ilat+1,ilon]  , lonh[ilat+1,ilon]    ],
            #                    [lath[ilat+1,ilon+1], lonh[ilat+1,ilon+1]  ],
            #                    [lath[ilat,ilon+1]  , lonh[ilat,  ilon+1]  ]]

            # tmp = shape_to_geometry(gridcell_edges, epsg=crs_caea)
            # tmp.SwapXY()              # switch lat/lon back
            # grid_cell_geom_gpd_wkt[ilat][ilon] = tmp

            # -------------------------
            # EPSG:3573   does not need a swap after transform ... and is much faster than transform with EPSG:3035
            # -------------------------
            #
            # Windows            Python 3.8.5 GDAL 3.1.3 --> lat/lon (Ming)
            # MacOS 10.15.6      Python 3.8.5 GDAL 3.1.3 --> lat/lon (Julie)
            # Graham             Python 3.8.2 GDAL 3.0.4 --> lat/lon (Julie)
            # Graham             Python 3.6.3 GDAL 2.2.1 --> lon/lat (Julie)
            # Ubuntu 18.04.2 LTS Python 3.6.8 GDAL 2.2.3 --> lon/lat (Etienne)
            #
            if osgeo_version < '3.0':
                gridcell_edges = [ [lonh[ilat,  ilon]   , lath[ilat,ilon]      ],            # for some reason need to switch lat/lon that transform works
                                   [lonh[ilat+1,ilon]   , lath[ilat+1,ilon]    ],
                                   [lonh[ilat+1,ilon+1] , lath[ilat+1,ilon+1]  ],
                                   [lonh[ilat,  ilon+1] , lath[ilat,ilon+1]    ]]
            else:
                gridcell_edges = [ [lath[ilat,ilon]     , lonh[ilat,  ilon]    ],            # for some reason lat/lon order works
                                   [lath[ilat+1,ilon]   , lonh[ilat+1,ilon]    ],
                                   [lath[ilat+1,ilon+1] , lonh[ilat+1,ilon+1]  ],
                                   [lath[ilat,ilon+1]   , lonh[ilat,  ilon+1]  ]]

            tmp = shape_to_geometry(gridcell_edges, epsg=crs_caea)
            grid_cell_geom_gpd_wkt[ilat][ilon] = tmp

elif ( Path(input_file).suffix == '.shp'):

    # -------------------------------
    # Grid-cells are actually polygons in a shapefile
    # -------------------------------
    print(' ')
    print('   (3) Extract shapes from shapefile ...')

    grid_cell_geom_gpd_wkt = [ [ [] for ilon in range(nlon) ] for ilat in range(nlat) ]   # nlat = nshapes, nlon = 1
    for ishape in range(nshapes):

        idx = np.where(model_grid_shp[key_colname_model] == ishape)[0]
        if len(idx) == 0:
            print("Polygon ID = {} not found in '{}'. Numbering of shapefile attribute '{}' needs to be [0 ... {}-1].".format(ishape,input_file,key_colname_model,nshapes))
            raise ValueError('Polygon ID not found.')
        if len(idx) > 1:
            print("Polygon ID = {} found multiple times in '{}' but needs to be unique. Numbering of shapefile attribute '{}' needs to be [0 ... {}-1].".format(ishape,input_file,key_colname_model,nshapes))
            raise ValueError('Polygon ID not unique.')
        idx  = idx[0]
        poly = model_grid_shp.loc[idx].geometry
        try:
            grid_cell_geom_gpd_wkt[ishape][0] = ogr.CreateGeometryFromWkt(poly.to_wkt())
        except:
            grid_cell_geom_gpd_wkt[ishape][0] = ogr.CreateGeometryFromWkt(poly.wkt)

else:

    print("File extension found: {}".format(input_file.split('.')[-1]))
    raise ValueError('Input file needs to be either NetCDF (*.nc) or a Shapefile (*.shp).')

# -------------------------------
# Derive overlay and calculate weights
# -------------------------------
print(' ')
print('   (4) Deriving weights ...')

filename = output_file
ff       = open(filename,'w')
ff.write(':GridWeights                     \n')
ff.write('   #                                \n')
ff.write('   # [# HRUs]                       \n')
ff.write('   :NumberHRUs       {0}            \n'.format(nsubbasins))
ff.write('   :NumberGridCells  {0}            \n'.format(nshapes))
ff.write('   #                                \n')
ff.write('   # [HRU ID] [Cell #] [w_kl]       \n')

error_dict = {}
for ikk,kk in enumerate(keys):

    ibasin = shape.loc[kk]

    area_basin = coord_catch_wkt[kk].Area()
    enve_basin = coord_catch_wkt[kk].GetEnvelope()   # bounding box around basin (for easy check of proximity)

    area_all = 0.0
    ncells   = 0

    data_to_write = []
    for ilat in range(nlat):
        for ilon in range(nlon):

            enve_gridcell  = grid_cell_geom_gpd_wkt[ilat][ilon].GetEnvelope()   # bounding box around grid-cell (for easy check of proximity)
            grid_is_close  = check_proximity_of_envelops(enve_gridcell, enve_basin)

            if grid_is_close: # this check decreases runtime DRASTICALLY (from ~6h to ~1min)

                grid_cell_area = grid_cell_geom_gpd_wkt[ilat][ilon].Area()

                inter = (grid_cell_geom_gpd_wkt[ilat][ilon].Buffer(0.0)).Intersection(coord_catch_wkt[kk].Buffer(0.0)) # "fake" buffer to avoid invalid polygons and weirdos dumped by ArcGIS
                area_intersect = inter.Area()

                area_all += area_intersect
                if area_intersect > 0:

                    ncells += 1
                    data_to_write.append( [int(ibasin[key_colname]),ilat,ilon,ilat*nlon+ilon,area_intersect/area_basin] )

    # mismatch between area of subbasin (shapefile) and sum of all contributions of grid cells (model output)
    error = (area_basin - area_all)/area_basin


    if abs(error) > area_error_threshold and area_basin > 500000.:
        # record all basins with errors larger 5% (if basin is larger than 0.5 km2)
        error_dict[int(ibasin[key_colname])] = [ error, area_basin ]
        for idata in data_to_write:
            print("   >>> {0},{1},{2},{3},{4}".format(idata[0],idata[1],idata[2],idata[3],idata[4]))
            ff.write("   {0}   {1}   {2}\n".format(idata[0],idata[3],idata[4]))

    else:
        # adjust such that weights sum up to 1.0
        for idata in data_to_write:
            corrected = idata[4] * 1.0/(1.0-error)
            print("   >>> {0},{1},{2},{3},{4}  (corrected to {5})".format(idata[0],idata[1],idata[2],idata[3],idata[4],corrected))
            ff.write("   {0}   {1}   {2}\n".format(idata[0],idata[3],corrected))

        if error < 1.0:
            area_all *= 1.0/(1.0-error)
        error    = 0.0

    print('   >>> (Sub-)Basin: {0} ({1} of {2})'.format(int(ibasin[key_colname]),ikk+1,nsubbasins))
    print('   >>> Derived area of {0}  cells: {1}'.format(ncells,area_all))
    print('   >>> Read area from shapefile:   {0}'.format(area_basin))
    print('   >>> error:                      {0}%'.format(error*100.))
    print('   ')

ff.write(':EndGridWeights \n')
ff.close()



# print out all subbasins that have large errors
if (error_dict != {}):
    print('')
    print('WARNING :: The following (sub-)basins show large mismatches between provided model')
    print('           grid and domains spefied in the shapefile. It seems that your model')
    print('           output is not covering the entire domain!')
    print("           { <basin-ID>: <error>, ... } = ")
    for attribute, value in error_dict.items():
        print('               {0} : {1:6.2f} % of {2:8.1f} km2 basin'.format(attribute, value[0]*100., value[1]/1000./1000.))
    print('')




print('')
print('Wrote: ',filename)
print('')
