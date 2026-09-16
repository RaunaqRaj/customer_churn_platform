import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from src.config import PROCESSED_DATA_DIR


def load_test_data():
    """Load the untouched test dataset."""

    X_test = pd.read_csv(
        PROCESSED_DATA_DIR / "X_test.csv"
    )

    y_test = pd.read_csv(
        PROCESSED_DATA_DIR / "y_test.csv"
    ).squeeze()

    return X_test, y_test


def train_final_model():
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


def evaluate_final_model(model, X_test, y_test):
    """Evaluate the final model on the test set."""

    probabilities = model.predict_proba(X_test)[:, 1]

    threshold = 0.40

    predictions = (
        probabilities >= threshold
    ).astype(int)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    print("\n" + "=" * 60)
    print("FINAL RANDOM FOREST EVALUATION")
    print("=" * 60)

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    return probabilities


def main():

    X_train = pd.read_csv(
        PROCESSED_DATA_DIR / "X_train.csv"
    )

    y_train = pd.read_csv(
        PROCESSED_DATA_DIR / "y_train.csv"
    ).squeeze()

    X_test, y_test = load_test_data()

    print("Training data:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("\nTest data:")
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    model = train_final_model()

    print("\nTraining final Random Forest...")

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")

    evaluate_final_model(
        model,
        X_test,
        y_test
    )


if __name__ == "__main__":
    main()