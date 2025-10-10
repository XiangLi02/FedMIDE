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
-- local1 ''
-- global1 ''
-- remain1 ''
- `--local1 ''` : Refers to the **federated dataset** used for collaborative training across clients.  
- `--global1 ''` : Refers to the **federated shared data**, representing global knowledge common to all clients.  
- `--remain1 ''` : Refers to the **personalized dataset**, containing client-specific data used for personalization.
```
