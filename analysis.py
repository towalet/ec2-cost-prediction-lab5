"""Analysis and regression specified in the EC2 activity."""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from data_preparation import FEATURE_COLUMNS


def cost_comparison(data):
    comparison = data[["Name", "API Name", "On Demand", "Linux Reserved cost"]].dropna().copy()
    comparison["Savings (USD/hour)"] = comparison["On Demand"] - comparison["Linux Reserved cost"]
    comparison["Savings (%)"] = 100 * comparison["Savings (USD/hour)"] / comparison["On Demand"].replace(0, float("nan"))
    return comparison.sort_values("On Demand")


def on_demand_outliers(data):
    q1, q3 = data["On Demand"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = data.loc[(data["On Demand"] < lower) | (data["On Demand"] > upper)]
    return outliers, lower, upper


def instance_family(data, family):
    return data.loc[data["API Name"].str.split(".").str[0].eq(family.lower())]


def train_model(data):
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURE_COLUMNS], data["On Demand"], test_size=0.2, random_state=42
    )
    model = LinearRegression().fit(x_train, y_train)
    predicted = model.predict(x_test)
    mse = mean_squared_error(y_test, predicted)
    metrics = {
        "MAE": mean_absolute_error(y_test, predicted),
        "MSE": mse,
        "RMSE": mse ** 0.5,
    }
    results = pd.DataFrame({"Actual": y_test, "Predicted": predicted})
    return model, metrics, results, len(x_train)


def predict_cost(model, memory, vcpus):
    inputs = pd.DataFrame([[memory, vcpus]], columns=FEATURE_COLUMNS)
    return float(model.predict(inputs)[0])
