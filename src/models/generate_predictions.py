import joblib
import pandas as pd

from src.config import MODEL_DIR, PROCESSED_DATA_DIR


def load_model():
    model_path = MODEL_DIR / "churn_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at: {model_path}"
        )

    return joblib.load(model_path)


def load_customer_data():
    dataset_path = (
        PROCESSED_DATA_DIR / "customer_ml_dataset.csv"
    )

    df = pd.read_csv(dataset_path)

    return df


def prepare_features(df):
    columns_to_drop = [
        "customer_id",
        "signup_date",
        "last_order_date",
        "tenure_months",
        "churn",
    ]

    X = df.drop(columns=columns_to_drop)

    X = pd.get_dummies(
        X,
        columns=["city", "plan_type"],
        drop_first=True
    )

    return X


def create_predictions(df, X, model):
    probabilities = model.predict_proba(X)[:, 1]

    result = df[
        [
            "customer_id",
            "age",
            "city",
            "plan_type",
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
            "avg_satisfaction_score",
        ]
    ].copy()

    result["churn_probability"] = probabilities

    result["risk_level"] = pd.cut(
        result["churn_probability"],
        bins=[-0.01, 0.30, 0.60, 1.00],
        labels=["Low", "Medium", "High"]
    )

    return result


def main():
    print("=" * 60)
    print("GENERATING CUSTOMER CHURN PREDICTIONS")
    print("=" * 60)

    df = load_customer_data()

    print("\nCustomer dataset:")
    print("Rows:", len(df))

    X = prepare_features(df)

    print("Feature matrix:", X.shape)

    model = load_model()

    predictions = create_predictions(
        df,
        X,
        model
    )

    output_path = (
        PROCESSED_DATA_DIR / "customer_predictions.csv"
    )

    predictions.to_csv(
        output_path,
        index=False
    )

    print("\nPredictions generated successfully!")

    print("\nOutput:")
    print(output_path)

    print("\nRisk distribution:")
    print(
        predictions["risk_level"]
        .value_counts()
        .sort_index()
    )

    print("\nSample:")
    print(
        predictions[
            [
                "customer_id",
                "churn_probability",
                "risk_level"
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()