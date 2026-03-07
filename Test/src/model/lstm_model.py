import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SEQUENCE_LENGTH, N_FEATURES, LSTM_UNITS_1, LSTM_UNITS_2,
    DENSE_UNITS, DROPOUT_RATE, LEARNING_RATE, RANDOM_SEED
)

# LSTM Model

tf.random.set_seed(RANDOM_SEED)

class LSTMModel:
    def __init__(self, sequence_length=SEQUENCE_LENGTH, n_features=N_FEATURES):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
    
    def build_model(self):
        model = keras.Sequential([
            layers.Input(shape=(self.sequence_length, self.n_features)),

            # first LSTM layer
            layers.LSTM(
                units=LSTM_UNITS_1,
                return_sequences=True,
                name='lstm_1'
            ),
            layers.Dropout(DROPOUT_RATE, name='dropout_1'),
            
            # second LSTM layer
            layers.LSTM(
                units=LSTM_UNITS_2,
                return_sequences=False,
                name='lstm_2'
            ),
            layers.Dropout(DROPOUT_RATE, name='dropout_2'),
            
            # fully connected dense layer
            layers.Dense(units=DENSE_UNITS, activation='relu', name='dense_1'),
            
            # output layer
            layers.Dense(units=1, activation='linear', name='output')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        
        print("=== LSTM Model Architecture ===")
        model.summary()
        print("=" * 60)
        
        return model
    
    def get_model(self):
        if self.model is None:
            raise ValueError("Model has not been built yet. Please call build_model() first.")
        return self.model
    
    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("Model has not been built yet.")
        
        self.model.save(filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath):
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from: {filepath}")
        return self.model
