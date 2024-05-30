import os, sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
#sys.path.append(os.getcwd())

'''
# For testing only
sys.argv += ["--watershed", "02BD002"]
sys.argv += ["--gauge_lat", "47.910556"]
sys.argv += ["--gauge_lon", "-84.743056"]
sys.argv += ["--experiment", "MDA1000_LA5"]
sys.argv += ["--start_date", "1980-01-01"]
sys.argv += ["--end_date", "1981-12-31"]
sys.argv += ["--save_forcing"]
'''

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--watershed", type = str, required=True)
parser.add_argument("--gauge_lat", type = str)
parser.add_argument("--gauge_lon", type = str)
parser.add_argument("--predictor_name", type = str, default = 'exp0')
parser.add_argument("--experiment", type = str, default = 'default')
parser.add_argument("--start_date", type = str, required=True)
parser.add_argument("--end_date", type = str, required=True)
parser.add_argument("--time_shift", type = str, default= '-5')
parser.add_argument("--save_forcing", action='store_true')
parser.add_argument("--routing_provided", action='store_true')
args = parser.parse_args()

project_root = Path(os.getcwd())

# Use Basinmaker to delineate the routing network
from routing_delineation import RoutingNetwork
routing_network = RoutingNetwork(args, project_root)
routing_network.run()


# Subbasin shapefile processing and generate grid weights
from subbasins_shapefiles import SubBasin
subbasin_shapefiles = SubBasin(args, project_root)
subbasin_shapefiles.run()


# Calculate static attributes of the watershed
from static_attributes import StaticAttributes
static_attributes = StaticAttributes(args, project_root)
static_attributes.run()


# Calculate the dynamic forcing of the watershed
os.system(f"python ./scripts/dynamic_forcing.py {args.watershed} {args.experiment} {str(project_root)} {args.start_date} {args.end_date} {args.time_shift}")


# Generating time series files for LSTM prediction
from lstm_preprocessing import Forcing2TimeSeries
forcing2ts = Forcing2TimeSeries(args, project_root)
forcing2ts.create_timeseries()
forcing2ts.config_training()


# Run the LSTM
os.chdir(os.getcwd() + '/model')
os.system("nh-schedule-runs evaluate --directory trained_model --runs-per-gpu 5 --gpu-ids 0 1")
os.system(f"nh-results-ensemble --run-dirs trained_model/{args.predictor_name}* --output-dir trained_model")
os.system(f"python ../scripts/ensemble2netcdf.py trained_model/test_ensemble_results.p trained_model/test_ensemble_results.nc {args.watershed}")
os.chdir(project_root)


# Processing raw LSTM result
from result_nc2csv import Result2CSV
result2csv = Result2CSV(args, project_root)
result2csv.run()


# Process the LSTM output for routing in Raven
from lstm2raven import LSTM2Raven
lstm2raven = LSTM2Raven(args, project_root)
lstm2raven.run()


# Calculate the metric from raven-routing
from raven_result_analysis import RoutingAnalysis
routeanalysis = RoutingAnalysis(args, project_root)
routeanalysis.run()
