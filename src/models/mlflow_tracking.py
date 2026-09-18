import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from src.data.prepare_ml_data import load_dataset, prepare_features
from sklearn.model_selection import train_test_split


# Connect to MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Create/select experiment
mlflow.set_experiment("Customer Churn Prediction")


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = load_dataset()

X, y = prepare_features(df)


# --------------------------------------------------
# 2. Same train/test split used in the project
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 3. Final tuned Random Forest parameters
# --------------------------------------------------

params = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_split": 10,
    "min_samples_leaf": 10,
    "random_state": 42,
}


# --------------------------------------------------
# 4. Start MLflow run
# --------------------------------------------------

with mlflow.start_run(run_name="Tuned Random Forest"):

    model = RandomForestClassifier(
        **params,
        n_jobs=-1
    )

    # Train model
    model.fit(X_train, y_train)

    # Get churn probabilities
    y_probability = model.predict_proba(X_test)[:, 1]

    # Project threshold
    threshold = 0.40

    y_pred = (y_probability >= threshold).astype(int)


    # --------------------------------------------------
    # 5. Calculate metrics
    # --------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    pr_auc = average_precision_score(
        y_test,
        y_probability
    )


    # --------------------------------------------------
    # 6. Log parameters
    # --------------------------------------------------

    mlflow.log_params(params)

    mlflow.log_param(
        "classification_threshold",
        threshold
    )


    # --------------------------------------------------
    # 7. Log metrics
    # --------------------------------------------------

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("pr_auc", pr_auc)


    # --------------------------------------------------
    # 8. Log trained model
    # --------------------------------------------------

    mlflow.sklearn.log_model(
    model,
    name="random_forest_model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)


    # --------------------------------------------------
    # 9. Display results
    # --------------------------------------------------

    print("\nMLflow run completed successfully!")

    print("\nModel Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")

    print("\nMLflow tracking completed!")