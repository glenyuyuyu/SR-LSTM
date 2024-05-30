from pathlib import Path
import os

project_root = Path(os.getcwd())

'''
# Run on single basin
os.system(f"python ./scripts/main.py  --watershed 08LB020\
          --predictor_name fraser\
          --experiment default\
          --start_date 1980-01-01\
          --end_date 2018-12-30\
          --time_shift -8\
          --save_forcing\
          --routing_provided")
'''

# Run on mutiple basins
basins = ['08JE001','08KA005']

for b in basins:
    os.system(f"python ./scripts/main.py  --watershed {b}\
          --predictor_name fraser\
          --experiment default\
          --start_date 2000-01-01\
          --end_date 2018-12-30\
          --time_shift -8\
          --save_forcing\
          --routing_provided")