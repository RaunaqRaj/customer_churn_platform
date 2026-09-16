import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from src.config import PROCESSED_DATA_DIR


def load_data():
    """Load training features and target."""

    X_train = pd.read_csv(
        PROCESSED_DATA_DIR / "X_train.csv"
    )

    y_train = pd.read_csv(
        PROCESSED_DATA_DIR / "y_train.csv"
    ).squeeze()

    return X_train, y_train


def get_models():
    """Create baseline models for comparison."""

    models = {

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ]),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        )
    }

    return models


def compare_models(X_train, y_train, models):
    """Compare models using 5-fold stratified cross-validation."""

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision"
    }

    results = []

    for name, model in models.items():

        print(f"\nEvaluating {name}...")

        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        row = {
            "Model": name
        }

        for metric in scoring:

            scores = cv_results[
                f"test_{metric}"
            ]

            row[metric.upper()] = scores.mean()

            row[f"{metric.upper()}_STD"] = scores.std()

        results.append(row)

    return pd.DataFrame(results)


def main():

    print("=" * 60)
    print("CUSTOMER CHURN MODEL COMPARISON")
    print("=" * 60)

    X_train, y_train = load_data()

    print("\nTraining data:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    models = get_models()

    results = compare_models(
        X_train,
        y_train,
        models
    )

    results = results.sort_values(
        by="PR_AUC",
        ascending=False
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)

    print(
        results[
            [
                "Model",
                "ACCURACY",
                "PRECISION",
                "RECALL",
                "F1",
                "ROC_AUC",
                "PR_AUC"
            ]
        ].round(4).to_string(index=False)
    )

    output_path = (
        PROCESSED_DATA_DIR /
        "model_comparison.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print("\nResults saved to:")
    print(output_path)


if __name__ == "__main__":
    main()