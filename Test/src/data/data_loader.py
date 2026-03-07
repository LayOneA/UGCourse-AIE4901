import pandas as pd
import os
from datetime import datetime, timedelta
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TIME_SLOTS_PER_DAY

# Data loader module

class DataLoader:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
    
    def load_date_range(self, start_date, end_date):
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        all_data = []
        current = start
        
        while current <= end:
            date_str = current.strftime('%Y-%m-%d')
            file_path = os.path.join(self.data_dir, f'{date_str}.csv')
            
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['date'] = date_str
                all_data.append(df)
                print(f"Loaded: {date_str}")
            else:
                print(f"File not found: {file_path}")
            
            current += timedelta(days=1)
        
        if not all_data:
            raise ValueError(f"No data files found between {start_date} and {end_date}")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\nA total of {len(combined_df)} transaction records were loaded")
        return combined_df
    
    def load_single_day(self, date):
        file_path = os.path.join(self.data_dir, f'{date}.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_csv(file_path)
        df['date'] = date
        return df
