import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEQUENCE_LENGTH, N_FEATURES

# Sequence generation module

class SequenceGenerator:
    def __init__(self, sequence_length=SEQUENCE_LENGTH):
        self.sequence_length = sequence_length
    
    def create_sequences(self, df):
        feature_columns = ['transaction_count', 'is_workday', 'day_of_week', 'time_slot']
        data = df[feature_columns].values
        
        X = []
        y = []
        metadata = []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length, 0])
            metadata.append({
                'date': df.iloc[i + self.sequence_length]['date'],
                'time_slot': df.iloc[i + self.sequence_length]['time_slot']
            })
        
        X = np.array(X)
        y = np.array(y)
                
        return X, y, metadata
    
    def split_by_date(self, df, train_dates, val_dates):
        train_df = df[df['date'].isin(train_dates)].reset_index(drop=True)
        val_df = df[df['date'].isin(val_dates)].reset_index(drop=True)

        X_train, y_train, meta_train = self.create_sequences(train_df)
        X_val, y_val, meta_val = self.create_sequences(val_df)
        
        return X_train, y_train, meta_train, X_val, y_val, meta_val
    
    def create_sequences_for_prediction(self, df, start_idx=0):
        feature_columns = ['transaction_count', 'is_workday', 'day_of_week', 'time_slot']
        data = df[feature_columns].values
        
        X = []
        metadata = []
        
        for i in range(start_idx, len(data) - self.sequence_length + 1):
            X.append(data[i:i + self.sequence_length])
            
            if i + self.sequence_length < len(df):
                metadata.append({
                    'date': df.iloc[i + self.sequence_length]['date'],
                    'time_slot': df.iloc[i + self.sequence_length]['time_slot']
                })
            else:
                metadata.append({
                    'date': df.iloc[-1]['date'],
                    'time_slot': (df.iloc[-1]['time_slot'] + 1) % 288
                })
        
        X = np.array(X)
        
        return X, metadata
