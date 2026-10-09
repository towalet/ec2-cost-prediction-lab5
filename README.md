# EC2 Cost Prediction Lab 5

Streamlit dashboard using the previous activity's 861-row EC2 dataset. It includes cost summaries, boxplots, IQR outliers, Reserved comparisons, T2/T3 analysis, and On-Demand cost prediction from memory and vCPUs.

## Run locally

Use Python 3.12. From this folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Test

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_lab.py
.\.venv\Scripts\python.exe -m pip check
```

## Deploy on Streamlit Community Cloud

1. Put app.py, analysis.py, data_preparation.py, ec2dataset.csv, and requirements.txt in the root of a GitHub repository.
2. Sign in at https://share.streamlit.io and choose Create app, then deploy from GitHub.
3. Select the repository, main branch, and app.py. In Advanced settings, select Python 3.12.
4. Deploy, then open the public URL and check all three tabs and both prediction inputs.

Official instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Mandatory demo

1. Open the deployed dashboard. On Dataset, show 861 original rows, 11 columns, and 841 regression rows. Explain that 20 rows lack On-Demand prices.
2. On Cost analysis, show the summaries, boxplots, IQR outliers, Reserved savings, and switch between T2 and T3.
3. On Prediction, explain that LinearRegression uses memory and vCPUs with an 80/20 split: 672 training and 169 test rows, random_state=42.
4. Show MAE, MSE, RMSE, and the actual-versus-predicted plot.
5. Use 4 GiB and 2 vCPUs: the raw prediction is -$0.3993/hour. Explain that a negative cost is invalid and demonstrates a limitation of this simple model.
6. Change to 64 GiB and 16 vCPUs. Show that the prediction updates to a positive value. Explain that these are dataset-based estimates, not live AWS quotes.

The mandatory demo must still be presented by the student. Track completion in CHECKLIST.md.
