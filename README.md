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

The underlying principle of this prediction logic is as follows: the real-time foot traffic of the canteen is essentially the result of continuous accumulation of the difference between the number of incoming diners and outgoing diners in each time slot. Meanwhile, the number of outgoing diners within a single time slot can be statistically calculated using the incoming diner data from multiple preceding time slots, combined with the mathematical distribution of users' dining duration.

It follows that the core of the entire prediction task lies in the statistics and prediction of incoming diners in each time slot, which can be equivalently regarded as the number of people completing transactions at the service window.

**Time slot Definition**: We divide the full 24-hour day into 288 time slots at 5-minute intervals, e.g., `Time Slot 1` covers 00:00–00:04.

## Code Structure

