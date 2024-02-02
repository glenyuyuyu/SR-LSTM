from pathlib import Path
import os

project_root = Path(os.getcwd())

# Run on single basin
os.system(f"python ./scripts/main.py  --watershed 02KB001\
          --experiment MDA200_LA5\
          --gauge_lat 45.886111\
          --gauge_lon -77.315278\
          --start_date 1980-01-01\
          --end_date 1981-12-31\
          --save_forcing")
