import os

# PATH CONFIGURATION
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'dataset', 'tipdm', 'cooked')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models')
REPORT_SAVE_DIR = os.path.join(BASE_DIR, 'reports')

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(REPORT_SAVE_DIR, exist_ok=True)

# DATASET CONFIGURATION
TRAIN_START_DATE = '2019-04-01'
TRAIN_END_DATE = '2019-04-23'

VAL_START_DATE = '2019-04-24'
VAL_END_DATE = '2019-04-30'

# TIME SLOT CONFIGURATION
TIME_SLOTS_PER_DAY = 288

# MODEL HYPERPARAMETERS
# Sequence configuration
SEQUENCE_LENGTH = 12 # 1 hour historical data
FEATURES = ['transaction_count', 'is_workday', 'day_of_week', 'time_slot']
N_FEATURES = len(FEATURES)

# LSTM Model configuration
LSTM_UNITS_1 = 128  # Number of units in the first LSTM layer
LSTM_UNITS_2 = 64   # Number of units in the second LSTM layer
DENSE_UNITS = 32    # Number of units in the dense layer
DROPOUT_RATE = 0.3  # Dropout rate

# Training configuration
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.0 # Use separate validation set

# Early Stopping configuration
PATIENCE = 15 
MIN_DELTA = 0.0001  

# Random seed
RANDOM_SEED = 42
