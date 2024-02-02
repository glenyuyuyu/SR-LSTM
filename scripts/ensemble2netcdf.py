# Load ensemble results file from argv[1], extract predictions,
# convert mm/day to m3/s, clip to non-negative values,
# and write all predictions into one NetCDF file at argv[2].
import pickle
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import xarray as xr

if __name__ == '__main__':
    results_file = pickle.load(open(Path(sys.argv[1]), 'rb'))
    basin_id = sys.argv[3]
    static_attributes = pd.read_csv(f'../model/attributes/{basin_id}_static_attributes.csv', index_col=[0])

    all_results = []
    for basin, basin_results in tqdm(results_file.items()):
        xr_results = basin_results['1D']['xr']

        # clip values to >= 0
        xr_results['qobs_mm_per_day_sim'] = xr.where(xr_results['qobs_mm_per_day_sim'] < 0,
                                                     0, xr_results['qobs_mm_per_day_sim'])
        
        area = static_attributes.loc[basin, 'area_km2'] * 1000_000  # km2 -> m2
        xr_results = xr_results * area / (60 * 60 * 24 * 1000.0)  # mm/d -> m3/s
        xr_results = xr_results.rename({f'qobs_mm_per_day_{x}': f'qobs_m3_per_s_{x}'
                                        for x in ['obs', 'sim']})

        xr_results = xr_results.expand_dims('basin')
        xr_results['basin'] = [basin]
        all_results.append(xr_results)

    all_results = xr.merge(all_results)
    all_results.to_netcdf(Path(sys.argv[2]))

