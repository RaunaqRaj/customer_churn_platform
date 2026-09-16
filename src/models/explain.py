import pandas as pd
import shap

from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from src.config import PROCESSED_DATA_DIR


def load_data():
    """Load training and test datasets."""

    X_train = pd.read_csv(
        PROCESSED_DATA_DIR / "X_train.csv"
    )

    y_train = pd.read_csv(
        PROCESSED_DATA_DIR / "y_train.csv"
    ).squeeze()

    X_test = pd.read_csv(
        PROCESSED_DATA_DIR / "X_test.csv"
    )

    y_test = pd.read_csv(
        PROCESSED_DATA_DIR / "y_test.csv"
    ).squeeze()

    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train):
    """Train the final tuned Random Forest."""

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def calculate_shap_values(model, X_test):
    """Calculate SHAP values for the test dataset."""

    print("\nCalculating SHAP values...")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        X_test
    )

    return explainer, shap_values


def get_churn_shap_values(shap_values):
    """
    Extract SHAP values for the churn class.

    SHAP versions can return either:
    - a list of arrays
    - a 3D NumPy array
    """

    if isinstance(shap_values, list):

        return shap_values[1]

    if len(shap_values.shape) == 3:

        return shap_values[:, :, 1]

    return shap_values


def show_global_feature_importance(
    shap_values,
    X_test
):
    """Display global SHAP feature importance."""

    print("\n" + "=" * 60)
    print("GLOBAL SHAP FEATURE IMPORTANCE")
    print("=" * 60)

    churn_shap_values = get_churn_shap_values(
        shap_values
    )

    mean_abs_shap = (
        pd.DataFrame({
            "feature": X_test.columns,
            "importance": (
                abs(churn_shap_values)
                .mean(axis=0)
            )
        })
        .sort_values(
            "importance",
            ascending=False
        )
    )

    print(
        mean_abs_shap
        .head(15)
        .round(6)
        .to_string(index=False)
    )

    return mean_abs_shap

def create_shap_summary_plot(
    shap_values,
    X_test
):
    """Create and save a SHAP summary plot."""

    print("\nCreating SHAP summary plot...")

    churn_shap_values = get_churn_shap_values(
        shap_values
    )

    plt.figure(
        figsize=(10, 8)
    )

    shap.summary_plot(
        churn_shap_values,
        X_test,
        show=False
    )

    plt.title(
        "SHAP Summary Plot - Customer Churn"
    )

    plt.tight_layout()

    output_path = (
        PROCESSED_DATA_DIR /
        "shap_summary_plot.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nSHAP summary plot saved to:")
    print(output_path)

def explain_customer(
    model,
    explainer,
    X_test,
    customer_index=0
):
    """Explain one customer's churn prediction."""

    customer = X_test.iloc[[customer_index]]

    probability = model.predict_proba(
        customer
    )[0, 1]

    churn_shap_values = get_churn_shap_values(
        explainer.shap_values(X_test)
    )

    customer_shap = churn_shap_values[
        customer_index
    ]

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

    print("\n" + "=" * 60)
    print("INDIVIDUAL CUSTOMER EXPLANATION")
    print("=" * 60)

    print(
        f"\nCustomer index: {customer_index}"
    )

    print(
        f"Churn probability: {probability:.4f}"
    )

    print("\nTop factors influencing this prediction:")

    print(
        explanation[
            [
                "feature",
                "feature_value",
                "shap_value"
            ]
        ]
        .head(10)
        .round(4)
        .to_string(index=False)
    )

    print("\nInterpretation:")

    positive_factors = explanation[
        explanation["shap_value"] > 0
    ].head(5)

    negative_factors = explanation[
        explanation["shap_value"] < 0
    ].head(5)

    print("\nFactors pushing prediction toward CHURN:")

    if len(positive_factors) == 0:
        print("None")

    else:
        for _, row in positive_factors.iterrows():
            print(
                f"  ↑ {row['feature']}: "
                f"{row['feature_value']} "
                f"(SHAP {row['shap_value']:.4f})"
            )

    print("\nFactors pushing prediction away from CHURN:")

    if len(negative_factors) == 0:
        print("None")

    else:
        for _, row in negative_factors.iterrows():
            print(
                f"  ↓ {row['feature']}: "
                f"{row['feature_value']} "
                f"(SHAP {row['shap_value']:.4f})"
            )

    return explanation

def main():

    print("=" * 60)
    print("SHAP EXPLAINABILITY")
    print("=" * 60)

    X_train, y_train, X_test, y_test = load_data()

    print("\nTraining data:")
    print("X_train:", X_train.shape)

    print("\nTest data:")
    print("X_test:", X_test.shape)

    print("\nTraining final Random Forest...")

    model = train_model(
        X_train,
        y_train
    )

    print("Model trained successfully.")

    explainer, shap_values = calculate_shap_values(
        model,
        X_test
    )

    importance = show_global_feature_importance(
        shap_values,
        X_test
    )

    create_shap_summary_plot(
        shap_values,
        X_test
    )

    explain_customer(
        model,
        explainer,
        X_test,
        customer_index=0
    )

    # Save global feature importance.
    output_path = (
        PROCESSED_DATA_DIR /
        "shap_feature_importance.csv"
    )

    importance.to_csv(
        output_path,
        index=False
    )

    print("\nSHAP feature importance saved to:")
    print(output_path)

    print("\nTop 5 churn-related features:")

    print(
        importance
        .head(5)
        .round(6)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()