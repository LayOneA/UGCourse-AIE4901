import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TIME_SLOTS_PER_DAY

# Data preprocessing module

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.is_fitted = False
    
    def aggregate_by_timeslot(self, df):
        aggregated = df.groupby(['date', 'time_slot', 'is_workday', 'day_of_week']).size().reset_index(name='transaction_count')
        return aggregated
    
    def fill_missing_timeslots(self, df):
        dates = df['date'].unique()
        all_data = []
        for date in dates:
            date_data = df[df['date'] == date].copy()

            is_workday = date_data['is_workday'].iloc[0] if len(date_data) > 0 else 1
            day_of_week = date_data['day_of_week'].iloc[0] if len(date_data) > 0 else 0

            full_timeslots = pd.DataFrame({
                'date': [date] * TIME_SLOTS_PER_DAY,
                'time_slot': range(TIME_SLOTS_PER_DAY),
                'is_workday': [is_workday] * TIME_SLOTS_PER_DAY,
                'day_of_week': [day_of_week] * TIME_SLOTS_PER_DAY
            })

            merged = full_timeslots.merge(
                date_data[['date', 'time_slot', 'transaction_count']],
                on=['date', 'time_slot'],
                how='left'
            )
            merged['transaction_count'] = merged['transaction_count'].fillna(0)
            
            all_data.append(merged)
        
        result = pd.concat(all_data, ignore_index=True)
        result = result.sort_values(['date', 'time_slot']).reset_index(drop=True)

        return result
    
    def normalize_features(self, df, fit=True):
        df_normalized = df.copy()
        
        if fit:
            df_normalized['transaction_count'] = self.scaler.fit_transform(
                df[['transaction_count']]
            )
            self.is_fitted = True
            print("Fitted scaler")
        else:
            if not self.is_fitted:
                raise ValueError("Scaler must be fitted on training data first")
            df_normalized['transaction_count'] = self.scaler.transform(
                df[['transaction_count']]
            )
            print("Applied scaler")
        
        return df_normalized
    
    def inverse_transform(self, values):
        if not self.is_fitted:
            raise ValueError("Scaler not fitted")

        if len(values.shape) == 1:
            values = values.reshape(-1, 1)
        
        return self.scaler.inverse_transform(values)
    
    def process(self, df, fit=True):
        print("\nStarting data preprocessing...")

        df_agg = self.aggregate_by_timeslot(df)
        df_filled = self.fill_missing_timeslots(df_agg)
        df_normalized = self.normalize_features(df_filled, fit=fit)
        
        print("Preprocessing completed.\n")
        
        return df_normalized
