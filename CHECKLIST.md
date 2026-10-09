# Lab 5 checklist

Sources: Activity - Analysing EC2 Instances (1).docx and the lab instructions supplied in chat.

- [x] Step 1: Review requirements and locate the previous activity dataset.
  - Copied ec2dataset.csv from Assignment 3- EC2 Dashboard/ec2-eda-dashboard.
  - Verified 861 rows, 11 matching columns, and identical SHA-256 hashes for the source and copy.
- [x] Step 2: Set up dependencies, load the data, and clean prices, memory, and vCPUs. Check missing values and usable rows.
  - Created a Python 3.12 virtual environment in .venv and recorded the five installed package versions in requirements.txt. All imports and pip check passed.
  - Added data_preparation.py with reusable loading and cleaning. Checked the cleaning steps against the activity document.
  - Verified every converted value against the original 861-row CSV; the source remains unchanged.
  - Missing values: On Demand 20, Linux Reserved 24, Linux Spot Minimum 39, Windows On Demand 263, Windows Reserved 266, memory 0, vCPUs 0. All missing prices correspond to unavailable source values.
  - Regression data: 841 usable rows; excluded 20 rows with missing On-Demand prices. All predictors are positive and available target prices are nonnegative.
  - Rerun this step with `.\.venv\Scripts\python.exe data_preparation.py`.
- [x] Step 3: Cover the document's analysis: cost summaries, boxplots, On-Demand IQR outliers, On-Demand versus Reserved costs, and T2/T3 comparisons.
  - Cost analysis tab includes all required summaries and charts, 66 IQR outliers, 837 On-Demand/Reserved comparisons, and exact T2/T3 filters (7 instances each).
- [x] Step 4: Train LinearRegression using Instance Memory and vCPUs to predict On Demand. Use an 80/20 train/test split with random_state=42. Verify MAE, MSE, RMSE, and an actual-versus-predicted scatter plot.
  - 672 training rows and 169 test rows. MAE 1.8133, MSE 26.9395, RMSE 5.1903. Coefficients verified against an independent least-squares calculation.
- [x] Step 5: Build and locally check the Streamlit dashboard, including predictions from memory and vCPU inputs. Check the document's example of 4 GiB and 2 vCPUs.
  - All four automated tests passed, including changed inputs, boundaries, family filtering, cleaning, and regression metrics. Dependency check passed.
  - Browser review confirmed all three tabs and working predictions. Local HTTP health check returned ok.
  - 4 GiB / 2 vCPUs predicts -0.3993 USD/hour; the dashboard explains the model limitation. 64 GiB / 16 vCPUs predicts 0.5069 USD/hour.
- [ ] Step 6: Deploy on Streamlit Community Cloud and verify that the public app loads and predicts successfully.
  - Prepared the local Git repository on main with the app, dataset, requirements, tests, README, and checklist. The origin remote points to the GitHub repository below.
  - Created https://github.com/towalet/ec2-cost-prediction-lab5. Upload is pending the Chrome extension's file URL permission; Streamlit sign-in is also pending. No cloud deployment yet.
- [ ] Step 7: Prepare and perform the mandatory demo, then check the completed lab against the requirements.
  - Demo walkthrough prepared in README.md. Local requirements checked against both parts of the activity and the chat instructions. The student still needs to present the demo after deployment.

Deployment guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started

Update this checklist after each verified step. The mandatory demo remains incomplete until it has been performed.
