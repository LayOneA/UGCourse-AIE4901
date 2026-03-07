import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import REPORT_SAVE_DIR

# Report generator module

class ReportGenerator:    
    def __init__(self):
        self.report_data = []
        self.figures_dir = os.path.join(REPORT_SAVE_DIR, 'figures')
        os.makedirs(self.figures_dir, exist_ok=True)
    
    def _timeslot_to_time(self, timeslot):
        start_minutes = timeslot * 5
        end_minutes = start_minutes + 4
        start_hour = start_minutes // 60
        start_min = start_minutes % 60
        end_hour = end_minutes // 60
        end_min = end_minutes % 60
        return f"{start_hour:02d}:{start_min:02d}-{end_hour:02d}:{end_min:02d}"
    
    def generate_report(self, y_true, y_pred, metadata, preprocessor):
        print("\nGenerating Test Report...")
        
        y_true_original = preprocessor.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_original = preprocessor.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        report_data = []
        for i, meta in enumerate(metadata):
            timeslot = int(meta['time_slot'])
            error = y_pred_original[i] - y_true_original[i]
            abs_error = abs(error)
            sq_error = error ** 2
            
            report_data.append({
                'Date': meta['date'],
                'Time_Slot': timeslot,
                'Time_Period': self._timeslot_to_time(timeslot),
                'True_Volume': int(round(y_true_original[i])),
                'Predicted_Volume': int(round(y_pred_original[i])),
                'Error': int(round(error)),
                'Absolute_Error': int(round(abs_error)),
                'Squared_Error': sq_error
            })
        df_report = pd.DataFrame(report_data)
        return df_report
    
    def calculate_metrics(self, df_report):
        metrics = {
            'MSE': df_report['Squared_Error'].mean(),
            'RMSE': np.sqrt(df_report['Squared_Error'].mean()),
            'MAE': df_report['Absolute_Error'].mean(),
            'Total_Samples': len(df_report)
        }
        return metrics
    
    def calculate_daily_metrics(self, df_report):
        daily_metrics = df_report.groupby('Date').agg({
            'Squared_Error': 'mean',
            'Absolute_Error': 'mean',
            'True_Volume': 'sum',
            'Predicted_Volume': 'sum'
        }).reset_index()
        daily_metrics.columns = ['Date', 'MSE', 'MAE', 'True_Total_Volume', 'Predicted_Total_Volume']
        daily_metrics['RMSE'] = np.sqrt(daily_metrics['MSE'])
        daily_metrics['Total_Volume_Error'] = daily_metrics['Predicted_Total_Volume'] - daily_metrics['True_Total_Volume']
        daily_metrics = daily_metrics[['Date', 'MSE', 'RMSE', 'MAE', 'True_Total_Volume', 'Predicted_Total_Volume', 'Total_Volume_Error']]
        return daily_metrics
    
    def save_daily_detailed_report(self, df_report, date):
        filename = os.path.join(REPORT_SAVE_DIR, f'Detailed_Report_{date}.txt')

        mse = df_report['Squared_Error'].mean()
        mae = df_report['Absolute_Error'].mean()
        true_total = df_report['True_Volume'].sum()
        pred_total = df_report['Predicted_Volume'].sum()
        total_error = pred_total - true_total
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write(f"Detailed Report of {date}\n")
            f.write("=" * 100 + "\n\n")
            
            f.write(f"{'Time Slot':<12} {'Time Period':<20} {'True':<12} {'Predicted':<12} "
                   f"{'Error':<12} {'MSE':<12} {'MAE':<12}\n")
            f.write("-" * 100 + "\n")
            
            for _, row in df_report.iterrows():
                f.write(f"{row['Time_Slot']:<12} {row['Time_Period']:<20} "
                       f"{row['True_Volume']:<12} {row['Predicted_Volume']:<12} "
                       f"{row['Error']:<12} {row['Squared_Error']:<12.2f} {row['Absolute_Error']:<12}\n")
            
            f.write("\n" + "=" * 100 + "\n")
            f.write("Daily Summary\n")
            f.write("=" * 100 + "\n")
            f.write(f"MSE:                     {mse:.4f}\n")
            f.write(f"MAE:                     {mae:.4f}\n")
            f.write(f"True Total Volume:       {int(true_total)}\n")
            f.write(f"Predicted Total Volume:  {int(pred_total)}\n")
            f.write(f"Total Volume Error:      {int(total_error)}\n")
        
        return filename
    
    def plot_daily_prediction(self, df_report, date):
        plt.figure(figsize=(14, 6))
        
        plt.plot(df_report['Time_Slot'], df_report['True_Volume'], 
                label='True Volume', linewidth=2, marker='o', markersize=3)
        plt.plot(df_report['Time_Slot'], df_report['Predicted_Volume'], 
                label='Predicted Volume', linewidth=2, marker='s', markersize=3)
        
        plt.xlabel('Time Slot', fontsize=12)
        plt.ylabel('Transaction Volume', fontsize=12)
        plt.title(f'Prediction Result of {date}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = os.path.join(self.figures_dir, f'Prediction_{date}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def plot_daily_errors(self, df_report, date):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        ax1.plot(df_report['Time_Slot'], df_report['Squared_Error'], 
                color='red', linewidth=2, marker='o', markersize=3)
        ax1.set_xlabel('Time Slot', fontsize=12)
        ax1.set_ylabel('MSE', fontsize=12)
        ax1.set_title(f'Mean Squared Error by Time Slot - {date}', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        ax2.plot(df_report['Time_Slot'], df_report['Absolute_Error'], 
                color='orange', linewidth=2, marker='s', markersize=3)
        ax2.set_xlabel('Time Slot', fontsize=12)
        ax2.set_ylabel('MAE', fontsize=12)
        ax2.set_title(f'Mean Absolute Error by Time Slot - {date}', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        filename = os.path.join(self.figures_dir, f'Errors_{date}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return filename
    
    def plot_overall_metrics(self, daily_metrics, start_date, end_date):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        dates = daily_metrics['Date'].values
        x_pos = np.arange(len(dates))
        
        # MSE plot
        axes[0, 0].bar(x_pos, daily_metrics['MSE'], 
                       color='red', alpha=0.8)
        axes[0, 0].set_xlabel('Date', fontsize=12)
        axes[0, 0].set_ylabel('MSE', fontsize=12)
        axes[0, 0].set_title('Daily MSE', fontsize=14, fontweight='bold')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(dates, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # RMSE plot
        axes[0, 1].bar(x_pos, daily_metrics['RMSE'], 
                       color='blue', alpha=0.8)
        axes[0, 1].set_xlabel('Date', fontsize=12)
        axes[0, 1].set_ylabel('RMSE', fontsize=12)
        axes[0, 1].set_title('Daily RMSE', fontsize=14, fontweight='bold')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(dates, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # MAE plot
        axes[1, 0].bar(x_pos, daily_metrics['MAE'], 
                       color='orange', alpha=0.8)
        axes[1, 0].set_xlabel('Date', fontsize=12)
        axes[1, 0].set_ylabel('MAE', fontsize=12)
        axes[1, 0].set_title('Daily MAE', fontsize=14, fontweight='bold')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(dates, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Volume comparison
        bar_width = 0.35
        axes[1, 1].bar(x_pos - bar_width/2, daily_metrics['True_Total_Volume'], 
                      bar_width, label='True Volume', alpha=0.8)
        axes[1, 1].bar(x_pos + bar_width/2, daily_metrics['Predicted_Total_Volume'], 
                      bar_width, label='Predicted Volume', alpha=0.8)
        axes[1, 1].set_xlabel('Date', fontsize=12)
        axes[1, 1].set_ylabel('Total Volume', fontsize=12)
        axes[1, 1].set_title('Daily Total Volume Comparison', fontsize=14, fontweight='bold')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(dates, rotation=45, ha='right')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        filename = os.path.join(self.figures_dir, f'Overall_Metrics_from_{start_date}_to_{end_date}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def save_overall_report(self, daily_metrics, overall_metrics, start_date, end_date):
        filename = os.path.join(REPORT_SAVE_DIR, f'Overall_Report_from_{start_date}_to_{end_date}.txt')
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("LSTM Campus Cafeteria Traffic Prediction - Overall Test Report\n")
            f.write("=" * 100 + "\n\n")
            
            f.write("【Overall Evaluation Metrics】\n")
            f.write("-" * 100 + "\n")
            for key, value in overall_metrics.items():
                if isinstance(value, float):
                    f.write(f"{key:<25}: {value:.4f}\n")
                else:
                    f.write(f"{key:<25}: {value}\n")
            
            f.write("\n" + "=" * 100 + "\n")
            f.write("【Daily Evaluation Metrics】\n")
            f.write("=" * 100 + "\n\n")
            
            f.write(f"{'Date':<12} {'MSE':<12} {'RMSE':<12} {'MAE':<12} "
                   f"{'True Total':<15} {'Pred Total':<15} {'Total Error':<15}\n")
            f.write("-" * 100 + "\n")
            
            for _, row in daily_metrics.iterrows():
                f.write(f"{row['Date']:<12} {row['MSE']:<12.4f} {row['RMSE']:<12.4f} {row['MAE']:<12.4f} "
                       f"{int(row['True_Total_Volume']):<15} {int(row['Predicted_Total_Volume']):<15} "
                       f"{int(row['Total_Volume_Error']):<15}\n")
        
        print(f"Overall report saved: {filename}")
        return filename
    
    def generate_all_reports(self, df_report, daily_metrics, overall_metrics):
        generated_files = {
            'daily_predictions_viz': [],
            'daily_detailed': [],
            'daily_errors_viz': [],
            'overall_report': None,
            'overall_viz': None
        }
        
        print("\nGenerating All Reports...")
        
        dates = sorted(df_report['Date'].unique())
        start_date = dates[0]
        end_date = dates[-1]
        
        for date in dates:
            print(f"\nProcessing date: {date}")
            daily_data = df_report[df_report['Date'] == date]
            
            pred_viz = self.plot_daily_prediction(daily_data, date)
            generated_files['daily_predictions_viz'].append(pred_viz)
            print(f"-Daily prediction visualization: {os.path.basename(pred_viz)}")
            
            detail_file = self.save_daily_detailed_report(daily_data, date)
            generated_files['daily_detailed'].append(detail_file)
            print(f"-Daily detailed report: {os.path.basename(detail_file)}")
            
            error_viz = self.plot_daily_errors(daily_data, date)
            generated_files['daily_errors_viz'].append(error_viz)
            print(f"-Daily error visualization: {os.path.basename(error_viz)}")
        
        print("\nGenerating overall visualizations...")
        overall_viz = self.plot_overall_metrics(daily_metrics, start_date, end_date)
        generated_files['overall_viz'] = overall_viz
        print(f"-Overall metrics visualization: {os.path.basename(overall_viz)}")
        
        overall_report = self.save_overall_report(daily_metrics, overall_metrics, start_date, end_date)
        generated_files['overall_report'] = overall_report
        print(f"-Overall report: {os.path.basename(overall_report)}")
        
        return generated_files
    def print_summary(self, overall_metrics, daily_metrics):

        print("\n" + "=" * 80)
        print("Test Results Summary")
        print("=" * 80)
        
        print("\n【Overall Evaluation Metrics】")
        for key, value in overall_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        print("\n【Daily Evaluation Metrics】")
        print(daily_metrics.to_string(index=False))
        
        print("\n" + "=" * 80)
