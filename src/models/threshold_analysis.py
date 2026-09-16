import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DATA_DIR


def load_training_data():
    """Load training features and target."""

    X_train = pd.read_csv(
        PROCESSED_DATA_DIR / "X_train.csv"
    )

    y_train = pd.read_csv(
        PROCESSED_DATA_DIR / "y_train.csv"
    ).squeeze()

    return X_train, y_train


def create_model():
    """Create the tuned Random Forest model."""

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )

    return model


def analyze_business_thresholds(
    model,
    X_validation,
    y_validation
):
    """Analyze threshold performance from a business perspective."""

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    thresholds = [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50
    ]

    results = []

    total_customers = len(y_validation)
    actual_churners = int(y_validation.sum())

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        true_positives = int(
            ((predictions == 1) & (y_validation == 1)).sum()
        )

        false_positives = int(
            ((predictions == 1) & (y_validation == 0)).sum()
        )

        true_negatives = int(
            ((predictions == 0) & (y_validation == 0)).sum()
        )

        false_negatives = int(
            ((predictions == 0) & (y_validation == 1)).sum()
        )

        customers_flagged = int(
            predictions.sum()
        )

        precision = (
            true_positives /
            customers_flagged
            if customers_flagged > 0
            else 0
        )

        recall = (
            true_positives /
            actual_churners
            if actual_churners > 0
            else 0
        )

        f1 = (
            2 * precision * recall /
            (precision + recall)
            if precision + recall > 0
            else 0
        )

        flagged_percentage = (
            customers_flagged /
            total_customers
        ) * 100

        results.append({
            "threshold": threshold,
            "customers_flagged": customers_flagged,
            "flagged_percentage": flagged_percentage,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    return pd.DataFrame(results)


def main():

    print("=" * 70)
    print("BUSINESS-ORIENTED THRESHOLD ANALYSIS")
    print("=" * 70)

    X_train, y_train = load_training_data()

    print("\nOriginal training data:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    # Split training data into model-training and validation sets.
    X_model, X_validation, y_model, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train
    )

    print("\nModel training data:")
    print("X_model:", X_model.shape)
    print("y_model:", y_model.shape)

    print("\nValidation data:")
    print("X_validation:", X_validation.shape)
    print("y_validation:", y_validation.shape)

    print(
        "\nActual churners in validation set:",
        int(y_validation.sum())
    )

    # Train model.
    model = create_model()

    print("\nTraining Random Forest...")

    model.fit(
        X_model,
        y_model
    )

    print("Training complete.")

    # Analyze thresholds.
    results = analyze_business_thresholds(
        model,
        X_validation,
        y_validation
    )

    print("\n" + "=" * 70)
    print("BUSINESS THRESHOLD RESULTS")
    print("=" * 70)

    display_columns = [
        "threshold",
        "customers_flagged",
        "flagged_percentage",
        "true_positives",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
        "f1"
    ]

    print(
        results[display_columns]
        .round(4)
        .to_string(index=False)
    )

    # Save results.
    output_path = (
        PROCESSED_DATA_DIR /
        "business_threshold_analysis.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print("\nResults saved to:")
    print(output_path)


if __name__ == "__main__":
    main()