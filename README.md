# FEDMIDE: PERSONALIZED MEDICAL DATA IDENTIFICATION VIA DYNAMIC ENROLLMENT FOR FEDERATED LLMS

## Overview
Personalized federated learning (PFL) offers a potential solution to the challenge of privacy sensitivity and high heterogeneity of medical data for Large Language Models (LLMs). However, existing PFL methods often rely on architectural modifications, which introduce additional computational and communication overhead while overlooking the intrinsic value of medical data. To address this issue, we propose Personalized Medical Data Identification via Dynamic Enrollment for Federated LLMs (FedMIDE), a data-driven federated instruction tuning method that achieves efficient personalization solely through data-level control. FedMIDE incorporates a dynamic data inclusion strategy to enhance personalization performance. Extensive experiments demonstrate that FedMIDE achieves the highest accuracy among compared methods, while maintaining excellent robustness and stability.


## Requirements

The code requires some dependencies (**Python 3.8**) as specified in `requirements.txt`.  
Please install them with:

```bash
pip install -r requirements.txt
```

## Data

We provide a **federated medical dataset**, which is processed through a **warm-up federated learning** phase for data selection.  
As a result, **personalized data** and **federated shared data** are obtained and stored in the [data](https://github.com/XiangLi02/FedMIDE/tree/main/data) folder.

```bash
-- local1  'Federated dataset'
-- global1  'Federated train dataset'
-- remain1  'Personalized dataset'
```

## Running

Example usage:

```bash
-- python main.py --global_model 'chavinlo/alpaca-native'\
      --data_path  "./data" \
      --output_dir  './lora-shepherd-7b/'\
      --num_communication_rounds 10 \
      --num_clients  10 \
      --train_on_inputs \
      --group_by_length
```
