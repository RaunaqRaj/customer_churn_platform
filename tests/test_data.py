import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


def test_ml_dataset_exists():
    file_path = PROCESSED_DATA_DIR / "customer_ml_dataset.csv"

    assert file_path.exists()


def test_ml_dataset_not_empty():
    file_path = PROCESSED_DATA_DIR / "customer_ml_dataset.csv"

    df = pd.read_csv(file_path)

    assert len(df) > 0


def test_customer_ids_are_unique():
    file_path = RAW_DATA_DIR / "customers.csv"

    df = pd.read_csv(file_path)

    assert df["customer_id"].is_unique


def test_churn_values_are_valid():
    file_path = PROCESSED_DATA_DIR / "customer_ml_dataset.csv"

    df = pd.read_csv(file_path)

    assert set(df["churn"].unique()).issubset({0, 1})


def test_required_features_exist():
    file_path = PROCESSED_DATA_DIR / "customer_ml_dataset.csv"

    df = pd.read_csv(file_path)

    required_columns = [
        "customer_id",
        "age",
        "total_orders",
        "total_revenue",
        "days_since_last_order",
        "payment_failure_rate",
        "total_tickets",
        "avg_satisfaction_score",
        "churn",
    ]

    for column in required_columns:
        assert column in df.columns