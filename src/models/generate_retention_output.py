import pandas as pd
import shap
import joblib

from src.config import MODEL_DIR, PROCESSED_DATA_DIR


def load_model():
    model_path = MODEL_DIR / "churn_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at: {model_path}"
        )

    return joblib.load(model_path)


def load_data():
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


def get_shap_values(explainer, X):
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        return shap_values[1]

    if len(shap_values.shape) == 3:
        return shap_values[:, :, 1]

    return shap_values


def get_primary_risk_factor(feature_values, feature_names):
    positive_features = []

    for feature, value in zip(feature_names, feature_values):
        if value > 0:
            positive_features.append(
                (feature, value)
            )

    positive_features.sort(
        key=lambda x: abs(x[1]),
        reverse=True
    )

    if not positive_features:
        return "No strong positive risk factor identified"

    feature = positive_features[0][0]

    risk_factor_map = {
        "days_since_last_order":
            "Long time since last order",

        "total_tickets":
            "High support ticket activity",

        "avg_satisfaction_score":
            "Customer satisfaction concern",

        "payment_failure_rate":
            "Payment failure activity",

        "orders_per_month":
            "Low order frequency",

        "total_orders":
            "Customer order activity",

        "total_revenue":
            "Customer revenue pattern",

        "revenue_per_month":
            "Monthly revenue pattern",

        "tenure_days":
            "Customer tenure",

        "avg_resolution_time":
            "Long support resolution time",

        "inactive_customer_flag":
            "Customer inactivity",
            "support_ticket_rate":
            "High support ticket rate",

        "age":
            "Customer age pattern",

        "total_payments":
            "Payment activity",
    }

    return risk_factor_map.get(
        feature,
        feature.replace("_", " ").title()
    )


def get_recommended_action(risk_factor):
    action_map = {
        "Long time since last order":
            "Send a personalized re-engagement offer",

        "High support ticket activity":
            "Customer-success follow-up for support concerns",

        "Customer satisfaction concern":
            "Review customer experience and satisfaction",

        "Payment failure activity":
            "Contact the customer regarding payment issues",

        "Low order frequency":
            "Send targeted product recommendations",

        "Customer inactivity":
            "Launch an inactive-customer reactivation campaign",

        "Customer tenure":
            "Offer a loyalty or retention benefit",

        "Long support resolution time":
            "Review unresolved support experience",

        "Customer order activity":
            "Monitor customer engagement",

        "Customer revenue pattern":
            "Review customer value and engagement",

        "Monthly revenue pattern":
            "Review customer value and engagement",

        "High support ticket rate":
            "Customer-success follow-up for support concerns",

        "Customer age pattern":
            "Review customer engagement and personalize communication",

        "Payment activity":
            "Review customer payment engagement",

    }

    return action_map.get(
        risk_factor,
        "Continue regular customer engagement"
    )


def main():

    print("=" * 60)
    print("GENERATING POWER BI RETENTION DATASET")
    print("=" * 60)

    df = load_data()

    print("\nCustomers:", len(df))

    X = prepare_features(df)

    print("Features:", X.shape)

    model = load_model()

    print("\nCreating SHAP explanations...")

    explainer = shap.TreeExplainer(model)

    shap_values = get_shap_values(
        explainer,
        X
    )

    feature_names = X.columns.tolist()

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

    primary_factors = []
    recommended_actions = []

    for row in shap_values:

        risk_factor = get_primary_risk_factor(
            row,
            feature_names
        )

        action = get_recommended_action(
            risk_factor
        )

        primary_factors.append(risk_factor)
        recommended_actions.append(action)

    result["primary_risk_factor"] = primary_factors
    result["recommended_action"] = recommended_actions

    output_path = (
        PROCESSED_DATA_DIR /
        "customer_retention_output.csv"
    )

    result.to_csv(
        output_path,
        index=False
    )

    print("\nRetention dataset generated successfully!")

    print("\nOutput:")
    print(output_path)

    print("\nRisk distribution:")
    print(
        result["risk_level"]
        .value_counts()
        .sort_index()
    )

    print("\nPrimary risk factors:")
    print(
        result["primary_risk_factor"]
        .value_counts()
        .head(10)
    )

    print("\nSample:")
    print(
        result[
            [
                "customer_id",
                "churn_probability",
                "risk_level",
                "primary_risk_factor",
                "recommended_action",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()