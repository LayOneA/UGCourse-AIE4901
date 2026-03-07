import pandas as pd
import os
from datetime import datetime

# Extracted target dataset from the data2.csv file

def preprocess_data2():
    input_file = r'..\..\dataset\tipdm\raw\data2.csv'
    output_dir = r'..\..\dataset\tipdm\cooked'
    
    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_csv(input_file, encoding='gbk')
    except:
        try:
            df = pd.read_csv(input_file, encoding='gb18030')
        except:
            df = pd.read_csv(input_file, encoding='utf-8')
    
    df = df[df['Type'] == '消费'].copy()
    canteen_list = ['第一食堂', '第二食堂', '第三食堂', '第四食堂', '第五食堂', '教师食堂']
    df = df[df['Dept'].isin(canteen_list)].copy()
    df = df[['Date', 'Dept']].copy()
    dept_translation = {
        '第一食堂': '1st canteen',
        '第二食堂': '2nd canteen',
        '第三食堂': '3rd canteen',
        '第四食堂': '4th canteen',
        '第五食堂': '5th canteen',
        '教师食堂': 'teacher canteen'
    }
    df['Dept'] = df['Dept'].map(dept_translation)
    df.rename(columns={'Date': 'transaction_time', 'Dept': 'window'}, inplace=True)
    df['transaction_time'] = pd.to_datetime(df['transaction_time'], format='%Y/%m/%d %H:%M')
    before_filter = len(df)
    df = df[~((df['transaction_time'].dt.hour == 0) & (df['transaction_time'].dt.minute == 0))].copy()
    df['is_workday'] = df['transaction_time'].dt.dayofweek.apply(lambda x: 0 if x >= 5 else 1)
    df['day_of_week'] = df['transaction_time'].dt.dayofweek
    df['time_slot'] = (df['transaction_time'].dt.hour * 60 + df['transaction_time'].dt.minute) // 5
    df = df.sort_values('transaction_time')
    df['date'] = df['transaction_time'].dt.date

    for date, group in df.groupby('date'):
        group = group.drop('date', axis=1)
        filename = f"{date}.csv"
        output_path = os.path.join(output_dir, filename)
        group.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\nA total of {len(df.groupby('date'))} files were generated.")

if __name__ == '__main__':
    preprocess_data2()
