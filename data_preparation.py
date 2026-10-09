"""Load and clean the EC2 dataset for the lab analysis and dashboard."""

from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).resolve().with_name("ec2dataset.csv")
COST_COLUMNS = [
    "On Demand",
    "Linux Reserved cost",
    "Linux Spot Minimum cost",
    "Windows On Demand cost",
    "Windows Reserved cost",
]
FEATURE_COLUMNS = ["Instance Memory", "vCPUs"]
MODEL_COLUMNS = FEATURE_COLUMNS + ["On Demand"]


def load_data():
    """Return raw data, cleaned data, and rows usable for regression."""
    raw = pd.read_csv(DATA_PATH)
    data = raw.copy()

    for column in COST_COLUMNS:
        values = (
            data[column].str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.replace("hourly", "", regex=False)
            .str.strip()
        )
        # Unavailable prices remain missing; they are not free instances.
        data[column] = pd.to_numeric(values, errors="coerce")

    memory = (
        data["Instance Memory"].str.replace(" GiB", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    data["Instance Memory"] = pd.to_numeric(memory, errors="coerce")
    data["vCPUs"] = pd.to_numeric(
        data["vCPUs"].str.extract(r"(\d+)", expand=False), errors="coerce"
    )

    # Only the target and predictors determine eligibility for regression.
    model_data = data.dropna(subset=MODEL_COLUMNS).copy()
    return raw, data, model_data


if __name__ == "__main__":
    raw, data, model_data = load_data()
    print(f"Original dataset: {len(raw)} rows, {len(raw.columns)} columns")
    print("\nMissing values after cleaning:")
    print(data[COST_COLUMNS + FEATURE_COLUMNS].isna().sum().to_string())
    print(f"\nRows usable for regression: {len(model_data)}")
    print(f"Rows excluded from regression: {len(data) - len(model_data)}")
    print("\nCleaned preview (memory in GiB, prices in USD/hour):")
    print(data[["API Name"] + MODEL_COLUMNS].head().to_string(index=False))
