import os
import sys
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    TRAIN_START_DATE, TRAIN_END_DATE, 
    VAL_START_DATE, VAL_END_DATE,
    RANDOM_SEED, REPORT_SAVE_DIR
)
from data.data_loader import DataLoader
from data.preprocessor import DataPreprocessor
from data.sequence_generator import SequenceGenerator
from model.lstm_model import LSTMModel
from model.trainer import ModelTrainer
from report_generator import ReportGenerator

# RANDOM SEED
np.random.seed(RANDOM_SEED)


def get_date_list(start_date, end_date):
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    date_list = []
    current = start
    while current <= end:
        date_list.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    return date_list


def main():
    print("====== LSTM Transaction Prediction System ======")

    # 1. load data
    print("\nStep 1: Loading data")
    loader = DataLoader()
    print(f"\nLoading training data: {TRAIN_START_DATE} to {TRAIN_END_DATE}")
    train_raw = loader.load_date_range(TRAIN_START_DATE, TRAIN_END_DATE)
    print(f"\nLoading validation data: {VAL_START_DATE} to {VAL_END_DATE}")
    val_raw = loader.load_date_range(VAL_START_DATE, VAL_END_DATE)

    # 2. data preprocessing
    print("\nStep 2: Data Preprocessing")
    preprocessor = DataPreprocessor()
    print("\nProcessing training data and validation data")
    train_processed = preprocessor.process(train_raw, fit=True)
    val_processed = preprocessor.process(val_raw, fit=False)
    
    # 3. generate sequences
    print("\nStep 3: Generating sequences")
    generator = SequenceGenerator()
    train_dates = get_date_list(TRAIN_START_DATE, TRAIN_END_DATE)
    val_dates = get_date_list(VAL_START_DATE, VAL_END_DATE)
    all_data = preprocessor.fill_missing_timeslots(
        preprocessor.aggregate_by_timeslot(
            loader.load_date_range(TRAIN_START_DATE, VAL_END_DATE)
        )
    )
    all_data_normalized = preprocessor.normalize_features(all_data, fit=False)

    print("\nGenerating training sequences and validation sequences")
    X_train, y_train, meta_train = generator.create_sequences(train_processed)
    X_val, y_val, meta_val = generator.create_sequences(val_processed)
    
    # 4. build and train model
    print("\nStep 4: Building and training LSTM model")
    lstm_model = LSTMModel()
    model = lstm_model.build_model()
    trainer = ModelTrainer(model)
    history = trainer.train(X_train, y_train, X_val, y_val)
    
    # 5. evaluate model
    print("\nStep 5: Evaluating model")

    print("\nEvaluating on validation set...")
    eval_results = trainer.evaluate(X_val, y_val)
    print("\nGenerating predictions...")
    y_pred = trainer.predict(X_val)
    
    # 6. generate test report
    print("\nStep 6: Generating test reports and visualizations")
    
    report_gen = ReportGenerator()
    df_report = report_gen.generate_report(
        y_true=y_val,
        y_pred=y_pred,
        metadata=meta_val,
        preprocessor=preprocessor
    )

    overall_metrics = report_gen.calculate_metrics(df_report)
    daily_metrics = report_gen.calculate_daily_metrics(df_report)

    report_gen.print_summary(overall_metrics, daily_metrics)

    print("\nGenerating all reports and visualizations...")
    generated_files = report_gen.generate_all_reports(
        df_report=df_report,
        daily_metrics=daily_metrics,
        overall_metrics=overall_metrics
    )
    
    # ========== Complete ==========
    print("====== All steps completed successfully! ======")
    
    print("\nGenerated Reports Summary:")
    print(f"  - Daily prediction visualizations: {len(generated_files['daily_predictions_viz'])} files")
    print(f"  - Daily detailed reports: {len(generated_files['daily_detailed'])} files")
    print(f"  - Daily error visualizations: {len(generated_files['daily_errors_viz'])} files")
    print(f"  - Overall report: 1 file")
    print(f"  - Overall metrics visualization: 1 file")
    
    print(f"\nAll reports saved in: {REPORT_SAVE_DIR}")
    print(f"All visualizations saved in: {os.path.join(REPORT_SAVE_DIR, 'figures')}")
    
    print("\n")
    print("Thank you for using the LSTM Transaction Prediction System!")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
    except Exception as e:
        print(f"\n\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
