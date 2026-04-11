# Smart Campus AI in Action

## Project Background
This is the code repository for the AI capstone project (Course Code: AIE4901) at The Chinese University of Hong Kong, Shenzhen.

In response to the prevalent and prominent issues of queuing congestion and excessive waiting time during peak lunch and dinner hours of campus canteens (around 12:00 noon and 18:00 in the evening), our team has proposed multiple technical solutions and developed matching campus canteen foot traffic prediction tools aimed at assisting faculty and students in rationally arranging their dining schedules through accurate prediction. These methods include traditional methods based on mathematical modeling, AI methods based on computer vision, and AI methods based on non-computer vision.

This repository contains the code and model implementation of the AI methods based on non-computer vision developed by our project team. At its core, the solution adopts the Long Short-Term Memory (LSTM) as its algorithmic backbone.

## Team Members
1. LI Wenbo(李文博)   ID:122090276 (Project Leader)
2. WEN Langxuan(温朗萱)   ID:122090570
3. YANG Jiahao(杨家豪)   ID:122090646 (Responsible for Non-CV method; Owner of this repository)
4. LEI Mingcong(雷鸣聪)  ID:122090249

## Data Privacy Statement
***Pursuant to the data privacy protection agreement, the real campus canteen transaction record data from CUHK-Shenzhen cannot be released to any third party outside the authorized project team.***

To share our technical solution in a compliant manner, we adopted publicly available datasets as a substitute. We made appropriate adjustments to the code structure to be compatible with the public datasets, while preserving the intact core logic of the code. This is the reason we have named this code repository the Test Version.

The public available dataset we selected is sourced from the official competition dataset of the TiPDM Cup. To access the dataset `(path: Test/datasets/tipdm/raw)`, you can download via these link:
- https://github.com/Nicole456/Analysis-of-students-consumption-behavior-on-campus/tree/master/data
- https://www.tipdm.org/jn2stysj/1619.jhtml

## Method Overview
Specifically, this Non-Computer Vision method develops a canteen transaction volume prediction system based on the LSTM model. The core objective of the system is to accurately predict the total transaction volume of the next time slot following the current one (for the detailed definition of time slot, please refer to the subsequent description).

**Time slot Definition**: We divide the full 24-hour day into 288 time slots at 5-minute intervals, e.g., `Time Slot 1` covers 00:00–00:04.

The underlying principle of this prediction logic is as follows: the real-time foot traffic of the canteen is essentially the result of continuous accumulation of the difference between the number of incoming diners and outgoing diners in each time slot. Meanwhile, the number of outgoing diners within a single time slot can be statistically calculated using the incoming diner data from multiple preceding time slots, combined with the mathematical distribution of users' dining duration.

It follows that the core of the entire prediction task lies in the statistics and prediction of incoming diners in each time slot, which can be equivalently regarded as the number of people completing transactions at the service window.

## Model Implementation
The project implementation is divided into four core modules, with details as follows:

- **Data Preprocessing and Feature Engineering**: First, we perform preprocessing on daily transaction records, complete data aggregation at the granularity of predefined time slots, and calculate the total transaction volume for each time slot. We then conduct feature engineering on the preprocessed data to enrich the information dimension of input features. Specifically, the test version integrated two key features: `is_workday` and `day_of_week`; the official version adds more campus-related features (e.g., `HolidayLabel`) on this basis.
- **Model Architecture Design**: The core model adopts a base architecture consisting of a two-layer LSTM network paired with fully connected layers. A built-in Dropout mechanism is included to mitigate model overfitting, and the model supports automated initialization based on preconfigured parameters.
- **Training Optimization Mechanism**: To improve the model's training performance, the training pipeline is equipped with supporting mechanisms including Early Stopping, optimal model snapshot saving, and adaptive learning rate scheduling, which can automatically screen and lock the globally optimal model parameters.
- **Inference Logic and Training Objective**: The model takes time series transaction volume data and corresponding features from the past 12 time slots (equivalent to 1 hour) as input, and outputs the predicted total transaction volume for the next time slot. The model uses Mean Squared Error (MSE) as the target loss function for training.

## Code Structure
The project directory is organized as follows:

```bash
Test
├─data
│  └─tipdm
│     └─datatrans.py            # Data transformation script
├─dataset
│  └─tipdm
│     ├─cooked
│     │   └─...                 # Preprocessed data files
│     └─raw
│         └─...                 # Original competition data
├─models
│  └─best_model.keras           # Best LSTM model checkpoint
├─reports
│  └─...                        # Model results and figures
├─src
│  ├─data
│  │  ├─data_loader.py          # Load transaction data
│  │  ├─preprocessor.py         # Data preprocessing
│  │  └─sequence_generator.py   # Generate LSTM sequences
│  ├─model
│  │  ├─lstm_model.py           # LSTM model definition
│  │  └─trainer.py              # Model training logic
│  ├─config.py                  # Project configuration
│  ├─main.py                    # Main workflow entry
│  └─report_generator.py        # Generate analysis reports
```
## Simulator Execution
Navigate to the `src` directory and run `main.py`:
```bash
cd src
python main.py
```
## Execution Results
After running the project, generated performance report will be saved in the `report` folder:
```bash
└─report
   ├─figures
   │  └─ ... # figure report
   └─ ... # text report
```
As this is a test version with limited data volume and suboptimal prediction accuracy, its relevant details will not be elaborated in this `README`. Interested user can refer to the code repository.

**Model Performance of the Official Version**

For the official model, we adopted historical canteen transaction record data officially provided by CUHK-Shenzhen as the training dataset, which covers a total of more than 600,000 valid transaction records.

In the model validation stage, we selected transaction data from 9 consecutive calendar days in March 2026 of a CUHK-Shenzhen campus canteen as the offline validation dataset. The validation results show that the actual total transaction volume of this canteen in this period was 21,191 transactions, while the model-predicted total transaction volume reached 20,055 transactions. The Mean ***Absolute Percentage Error (MAPE)*** of the prediction is as low as ***5.36%***, achieving good prediction accuracy.



