# Data: IBM Telco Customer Churn

| | |
|---|---|
| **File** | `Telco-Customer-Churn.csv` (7,043 rows × 21 columns) |
| **Publisher** | IBM (sample dataset for IBM Cognos Analytics / Watson Studio) |
| **Downloaded from** | [github.com/IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv) (IBM's official repository) |
| **Also on Kaggle** | [blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): same file |
| **SHA-256** | `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` |

`src/churn/data.py` verifies this checksum every time the data is loaded, and re-downloads the
file from IBM's repository if it is missing.

Each row is one customer of a fictional California telecom company in one quarter.

| Group | Columns |
|---|---|
| Target | `Churn`: customer left within the last month (Yes/No; 26.5% Yes) |
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account | `tenure` (months), `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| Services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |

**Known quirk:** `TotalCharges` is stored as text, and 11 customers with `tenure = 0` have a blank
value (they haven't been billed yet). These are set to 0.
