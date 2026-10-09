"""Streamlit dashboard for the EC2 cost analysis lab."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from analysis import cost_comparison, instance_family, on_demand_outliers, predict_cost, train_model
from data_preparation import COST_COLUMNS, FEATURE_COLUMNS, load_data


st.set_page_config(page_title="EC2 Cost Prediction", layout="wide")
st.title("EC2 Cost Prediction")
st.caption("Cloud Economics · Lab 5 · Prices from the previous activity dataset, in USD per hour")


@st.cache_data
def prepare():
    return load_data()


@st.cache_resource
def fit_model():
    return train_model(prepare()[2])


raw, data, model_data = prepare()
model, metrics, results, training_count = fit_model()
prediction_tab, analysis_tab, data_tab = st.tabs(["Prediction", "Cost analysis", "Dataset"])


def show_boxplot(frame, title):
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(data=frame[COST_COLUMNS], ax=ax, color="#a8cce3", showmeans=True)
    ax.set(title=title, ylabel="Cost (USD/hour)")
    ax.set_xticks(range(5), ["Linux\nOn-Demand", "Linux\nReserved", "Linux\nSpot minimum", "Windows\nOn-Demand", "Windows\nReserved"])
    fig.tight_layout()
    st.pyplot(fig, width=900)
    plt.close(fig)


with prediction_tab:
    st.subheader("Predict On-Demand cost")
    st.write("Enter memory and vCPUs to estimate Linux On-Demand hourly cost using linear regression.")
    left, right = st.columns(2)
    memory = left.number_input(
        "Instance memory (GiB)", min_value=float(model_data["Instance Memory"].min()),
        max_value=float(model_data["Instance Memory"].max()), value=4.0, step=0.5, key="memory",
    )
    vcpus = right.number_input(
        "vCPUs", min_value=int(model_data["vCPUs"].min()),
        max_value=int(model_data["vCPUs"].max()), value=2, step=1, key="vcpus",
    )
    prediction = predict_cost(model, memory, vcpus)
    st.metric("Predicted On-Demand cost (USD/hour)", f"${prediction:.4f}")
    if prediction < 0:
        st.warning("The linear model predicts a negative cost for this configuration. This is not a valid price estimate; it shows a limitation of this model. The raw prediction is shown without clipping.")
    st.caption("Estimates update when inputs change. This model uses only memory and vCPUs; instance family and other pricing factors are not included.")

    st.subheader("Model evaluation")
    st.write(f"80/20 split with random_state=42: {training_count} training rows and {len(results)} test rows.")
    columns = st.columns(3)
    for col, (label, value) in zip(columns, metrics.items()):
        col.metric(label, f"{value:.4f}")
    st.caption("MAE and RMSE are in USD/hour. MSE is in squared USD/hour. Lower values indicate smaller errors on the held-out test set.")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(results["Actual"], results["Predicted"], alpha=0.6, s=22)
    low = min(results.min().min(), 0)
    high = results.max().max()
    ax.plot([low, high], [low, high], "r--", label="Perfect prediction")
    ax.set(xlabel="Actual On-Demand cost (USD/hour)", ylabel="Predicted cost (USD/hour)", title="Actual vs predicted On-Demand costs")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, width=900)
    plt.close(fig)
    with st.expander("Regression equation and test predictions"):
        st.write(f"Cost = {model.intercept_:.6f} + {model.coef_[0]:.6f} × memory (GiB) + {model.coef_[1]:.6f} × vCPUs")
        st.dataframe(results, hide_index=True)

with analysis_tab:
    st.subheader("Cost summary")
    st.dataframe(data[COST_COLUMNS].describe().round(4))
    show_boxplot(data, "Hourly cost comparison")

    st.subheader("On-Demand outliers")
    outliers, lower, upper = on_demand_outliers(data)
    st.write(f"IQR method: {len(outliers)} potential outliers. Lower bound: {lower:.4f}; upper bound: {upper:.4f} USD/hour.")
    st.caption("These are potential cost outliers, not necessarily errors. They remain in the regression dataset, as specified in the activity.")
    with st.expander("View outlier instances"):
        st.dataframe(outliers[["Name", "API Name", "Instance Memory", "vCPUs", "On Demand"]], hide_index=True)

    st.subheader("On-Demand versus Reserved")
    comparison = cost_comparison(data)
    st.write("Ten lowest On-Demand prices with an available Linux Reserved price:")
    st.dataframe(comparison.head(10).round(4), hide_index=True)
    st.write(f"Across {len(comparison)} comparable instances, the median Reserved saving is {comparison['Savings (%)'].median():.1f}% relative to On-Demand.")

    st.subheader("T2 and T3 families")
    family = st.selectbox("Instance family", ["T2", "T3"], key="family")
    family_data = instance_family(data, family)
    st.write(f"{len(family_data)} {family} instances. Families are matched using the API Name.")
    st.dataframe(family_data[COST_COLUMNS].describe().round(4))
    show_boxplot(family_data, f"{family} hourly cost distribution")
    combined = pd.concat([instance_family(data, "T2"), instance_family(data, "T3")])
    st.write("Ten lowest On-Demand prices across T2 and T3:")
    st.dataframe(cost_comparison(combined).head(10).round(4), hide_index=True)
    st.write("Lower hourly price does not guarantee the best fit: compare memory and vCPUs with workload needs. Reserved savings should be weighed against the required commitment.")

with data_tab:
    st.subheader("Dataset overview")
    a, b, c = st.columns(3)
    a.metric("Original rows", len(raw))
    b.metric("Columns", len(raw.columns))
    c.metric("Rows used for regression", len(model_data))
    st.write(f"{len(data) - len(model_data)} rows excluded from regression because a required price or predictor is missing.")
    st.caption("Source: ec2dataset.csv from the previous EC2 activity. Unavailable prices remain missing. Price units, GiB labels, and vCPU descriptions are converted to numbers.")
    with st.expander("Original data and types"):
        st.dataframe(raw, hide_index=True)
        st.dataframe(raw.dtypes.astype(str).rename("Original type"))
    st.write("Missing values after cleaning:")
    st.dataframe(data[COST_COLUMNS + FEATURE_COLUMNS].isna().sum().rename("Missing values"))
    st.write("Cleaned dataset:")
    st.dataframe(data, hide_index=True)
