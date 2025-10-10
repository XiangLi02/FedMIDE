# FEDMIDE: PERSONALIZED MEDICAL DATA IDENTIFICATION VIA DYNAMIC ENROLLMENT FOR FEDERATED LLMS

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
-- local1 'federated dataset'
-- global1 'federated train dataset'
-- remain1 'personalized dataset'
```
