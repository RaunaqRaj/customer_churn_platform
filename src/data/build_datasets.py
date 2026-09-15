import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.database.connection import engine


def load_ml_dataset():
    """Load customer ML features from PostgreSQL."""
    query = """
        SELECT *
        FROM customer_ml_features
        ORDER BY customer_id;
    """

    df = pd.read_sql(query, engine)

    return df


def main():
    # Create processed data directory if it doesn't exist
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load ML dataset from PostgreSQL
    df = load_ml_dataset()

    # Create a feature indicating whether the customer
    # had at least one previous order
    df["has_previous_order"] = (
        df["last_order_date"].notna().astype(int)
    )

    # Customers with no previous order have no recency value.
    # We represent this situation using -1.
    df["days_since_last_order"] = (
        df["days_since_last_order"].fillna(-1)
    )
    df["tenure_months"] = (
    df["tenure_days"] / 30.44
    ).round(2)

    # Calculate average number of orders per month.
    # Customers with zero tenure are assigned 0.
    df["orders_per_month"] = (
    df["total_orders"] / df["tenure_months"].replace(0, 1)
    ).round(2)

    df["revenue_per_month"] = (
    df["total_revenue"] / df["tenure_months"].replace(0, 1)
    ).round(2)

    # Create a flag indicating whether the customer
# has experienced at least one failed payment.
    df["has_payment_failure"] = (
    df["failed_payments"] > 0
    ).astype(int)

    df["has_support_ticket"] = (
    df["total_tickets"] > 0
    ).astype(int)

    df["low_satisfaction_flag"] = (
    (df["avg_satisfaction_score"] > 0)
    & (df["avg_satisfaction_score"] < 2.5)
    ).astype(int)

    # Create a flag indicating whether the customer
# has been inactive for at least 90 days.
    df["inactive_customer_flag"] = (
    df["days_since_last_order"] >= 90
    ).astype(int)

    # Calculate average number of support tickets per month.
# Customers with zero tenure are assigned 0.
    df["support_ticket_rate"] = (
    df["total_tickets"] / df["tenure_months"].replace(0, 1)
    ).round(2)
    # Save processed ML dataset
    output_path = PROCESSED_DATA_DIR / "customer_ml_dataset.csv"

    df.to_csv(output_path, index=False)

    # Display basic information
    print("ML dataset created successfully!")

    print("\nShape:")
    print(df.shape)

    print("\nSaved to:")
    print(output_path)

    print("\nChurn distribution:")
    print(df["churn"].value_counts())

    print("\nMissing values:")
    print(df.isnull().sum().to_string())

    # Display numerical feature summary
    print("\nFeature summary:")
    print(
        df[
            [
                "age",
                "tenure_days",
                "total_orders",
                "total_revenue",
                "average_order_value",
                "days_since_last_order",
                "total_payments",
                "failed_payments",
                "payment_failure_rate",
                "total_tickets",
                "avg_resolution_time",
                "avg_satisfaction_score"
            ]
        ].describe().round(2)
    )


if __name__ == "__main__":
    main()