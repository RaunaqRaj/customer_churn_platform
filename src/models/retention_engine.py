import pandas as pd
import shap
import mlflow.sklearn

from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DATA_DIR

def get_test_customer_mapping():
    """
    Recreate the original train/test split so that
    customer IDs remain aligned with X_test.
    """

    dataset_path = (
        PROCESSED_DATA_DIR
        / "customer_ml_dataset.csv"
    )

    df = pd.read_csv(dataset_path)

    columns_to_drop = [
        "customer_id",
        "signup_date",
        "last_order_date",
        "tenure_months",
        "churn"
    ]

    X = df.drop(
        columns=columns_to_drop
    )

    y = df["churn"]

    X = pd.get_dummies(
        X,
        columns=["city", "plan_type"],
        drop_first=True
    )

    _, X_test, _, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    customer_ids = df.loc[
        X_test.index,
        "customer_id"
    ]

    return customer_ids

def load_model():
    """
    Load the registered churn model from MLflow Model Registry.
    """

    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000"
    )

    model_uri = (
        "models:/customer-churn-random-forest/1"
    )

    model = mlflow.sklearn.load_model(
        model_uri
    )

    return model

def load_training_data():
    """
    Load the exact feature matrices used by the ML pipeline.
    """

    X_train = pd.read_csv(
        PROCESSED_DATA_DIR / "X_train.csv"
    )

    y_train = pd.read_csv(
        PROCESSED_DATA_DIR / "y_train.csv"
    ).squeeze()

    X_test = pd.read_csv(
        PROCESSED_DATA_DIR / "X_test.csv"
    )

    return X_train, y_train, X_test

def create_explainer(model):
    """
    Create a SHAP TreeExplainer for the Random Forest model.
    """

    return shap.TreeExplainer(model)

def get_churn_shap_values(shap_values):
    """
    Extract SHAP values for the churn class (class 1).
    Handles different SHAP output formats.
    """

    if isinstance(shap_values, list):
        return shap_values[1]

    if len(shap_values.shape) == 3:
        return shap_values[:, :, 1]

    return shap_values

def analyze_customer(
    customer_id,
    model,
    explainer,
    X_test,
    customer_ids
):
    """
    Analyze one real customer from the test dataset.
    """

    matching_indices = customer_ids[
        customer_ids == customer_id
    ].index

    if len(matching_indices) == 0:
        raise ValueError(
            f"Customer {customer_id} was not found "
            "in the test dataset."
        )

    original_index = matching_indices[0]

    test_position = customer_ids.index.get_loc(
        original_index
    )

    customer = X_test.iloc[[test_position]]

    churn_probability = model.predict_proba(
        customer
    )[0, 1]

    customer_shap_values = explainer.shap_values(
    customer
    )

    churn_shap_values = get_churn_shap_values(
    customer_shap_values
    )

    customer_shap = churn_shap_values[0]

    explanation = pd.DataFrame({
        "feature": X_test.columns,
        "feature_value": customer.iloc[0].values,
        "shap_value": customer_shap
    })

    explanation["absolute_shap"] = (
        explanation["shap_value"].abs()
    )

    explanation = explanation.sort_values(
        "absolute_shap",
        ascending=False
    )

    return build_retention_result(
        customer_id=customer_id,
        churn_probability=churn_probability,
        explanation=explanation
    )

def get_risk_level(churn_probability):
    """
    Convert churn probability into a business risk level.
    """

    if churn_probability >= 0.60:
        return "High"

    elif churn_probability >= 0.30:
        return "Medium"

    else:
        return "Low"

def get_risk_factors(explanation, top_n=5):
    """
    Convert positive SHAP values into business-friendly
    churn risk factors.
    """

    positive_factors = explanation[
        explanation["shap_value"] > 0
    ].copy()

    positive_factors = positive_factors.sort_values(
        "shap_value",
        ascending=False
    )

    risk_factors = []

    for _, row in positive_factors.head(top_n).iterrows():

        feature = row["feature"]
        value = row["feature_value"]

        if feature == "days_since_last_order":
            reason = (
                f"Customer has not ordered for "
                f"{int(value)} days"
            )

        elif feature == "total_tickets":
            reason = (
                f"Customer has raised "
                f"{int(value)} support tickets"
            )

        elif feature == "avg_satisfaction_score":
            reason = (
                f"Customer satisfaction score is "
                f"{value:.1f}"
            )

        elif feature == "payment_failure_rate":
            reason = (
                f"Payment failure rate is "
                f"{value:.1f}%"
            )

        elif feature == "orders_per_month":
            reason = (
                f"Customer places approximately "
                f"{value:.2f} orders per month"
            )

        elif feature == "total_orders":
            reason = (
                f"Customer has placed "
                f"{int(value)} orders"
            )

        elif feature == "total_revenue":
            reason = (
                f"Customer's total revenue is "
                f"{value:.2f}"
            )

        elif feature == "revenue_per_month":
            reason = (
                f"Customer generates approximately "
                f"{value:.2f} revenue per month"
            )

        elif feature == "tenure_days":
            reason = (
                f"Customer tenure is "
                f"{int(value)} days"
            )

        elif feature == "avg_resolution_time":
            reason = (
                f"Average support resolution time is "
                f"{value:.1f} days"
            )

        else:
            reason = f"{feature}: {value}"

        risk_factors.append(reason)

    return risk_factors

def get_retention_action(explanation):
    """
    Recommend a retention action based on
    the customer's top churn-related factors.
    """

    positive_factors = explanation[
        explanation["shap_value"] > 0
    ].copy()

    features = set(
        positive_factors["feature"]
    )

    actions = []

    if "days_since_last_order" in features:
        actions.append(
            "Send a personalized re-engagement offer"
        )

    if "payment_failure_rate" in features:
        actions.append(
            "Contact the customer regarding payment issues"
        )

    if "total_tickets" in features:
        actions.append(
            "Customer-success follow-up for support concerns"
        )

    if "avg_satisfaction_score" in features:
        actions.append(
            "Review customer experience and satisfaction"
        )

    if "orders_per_month" in features:
        actions.append(
            "Send targeted product recommendations"
        )

    if "inactive_customer_flag" in features:
        actions.append(
            "Launch an inactive-customer reactivation campaign"
        )

    if "tenure_days" in features:
        actions.append(
            "Offer a loyalty or retention benefit"
        )

    if not actions:
        actions.append(
            "Continue regular customer engagement"
        )

    return actions

def build_retention_result(
    customer_id,
    churn_probability,
    explanation
):
    """
    Build a complete customer retention analysis.
    """

    risk_level = get_risk_level(
        churn_probability
    )

    risk_factors = get_risk_factors(
        explanation
    )

    recommended_actions = get_retention_action(
        explanation
    )

    result = {
        "customer_id": customer_id,
        "churn_probability": round(
            float(churn_probability),
            4
        ),
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "recommended_actions": recommended_actions
    }

    return result

if __name__ == "__main__":

    X_train, y_train, X_test = load_training_data()

    customer_ids = get_test_customer_mapping()

    model = load_model()

    explainer = create_explainer(
        model
    )

    customer_id = customer_ids.iloc[0]

    result = analyze_customer(
        customer_id=customer_id,
        model=model,
        explainer=explainer,
        X_test=X_test,
        customer_ids=customer_ids
    )

    print("\n" + "=" * 60)
    print("REAL CUSTOMER RETENTION ANALYSIS")
    print("=" * 60)

    print(
        f"\nCustomer ID: "
        f"{result['customer_id']}"
    )

    print(
        f"Churn Probability: "
        f"{result['churn_probability']:.2%}"
    )

    print(
        f"Risk Level: "
        f"{result['risk_level']}"
    )

    print("\nTop Risk Factors:")

    for factor in result["risk_factors"]:
        print(f"- {factor}")

    print("\nRecommended Actions:")

    for action in result["recommended_actions"]:
        print(f"- {action}")