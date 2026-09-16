import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.config import PROCESSED_DATA_DIR


def load_data():
    """Load training and testing datasets."""

    X_train = pd.read_csv(
        PROCESSED_DATA_DIR / "X_train.csv"
    )

    X_test = pd.read_csv(
        PROCESSED_DATA_DIR / "X_test.csv"
    )

    y_train = pd.read_csv(
        PROCESSED_DATA_DIR / "y_train.csv"
    ).squeeze()

    y_test = pd.read_csv(
        PROCESSED_DATA_DIR / "y_test.csv"
    ).squeeze()

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Train Logistic Regression model."""

    # Scale numerical features
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    # Create Logistic Regression model
    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    # Train the model
    model.fit(X_train_scaled, y_train)

    return model, scaler

def train_decision_tree(X_train, y_train):
    """Train a Decision Tree model."""

    model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model

def train_random_forest(X_train, y_train):
    """Train a Random Forest model."""

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model

def train_xgboost(X_train, y_train):
    """Train an XGBoost model."""

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    return model

def evaluate_model(model, scaler, X_test, y_test):
    """Evaluate the trained model."""

    # Scale test data using the scaler
    # fitted on training data.
    X_test_scaled = scaler.transform(X_test)

    # Class predictions: 0 or 1
    y_pred = model.predict(X_test_scaled)

    # Probability of churn
    y_probability = model.predict_proba(
        X_test_scaled
    )[:, 1]

    print("\n========== MODEL EVALUATION ==========")

    print(
        f"Accuracy:  "
        f"{accuracy_score(y_test, y_pred):.4f}"
    )

    print(
        f"Precision: "
        f"{precision_score(y_test, y_pred):.4f}"
    )

    print(
        f"Recall:    "
        f"{recall_score(y_test, y_pred):.4f}"
    )

    print(
        f"F1 Score:  "
        f"{f1_score(y_test, y_pred):.4f}"
    )

    print(
        f"ROC-AUC:   "
        f"{roc_auc_score(y_test, y_probability):.4f}"
    )

    print(
        f"PR-AUC:    "
        f"{average_precision_score(y_test, y_probability):.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

def evaluate_tree(model, X_test, y_test):
    """Evaluate the Decision Tree model."""

    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(X_test)[:, 1]

    print("\n========== DECISION TREE EVALUATION ==========")

    print(
        f"Accuracy:  "
        f"{accuracy_score(y_test, y_pred):.4f}"
    )

    print(
        f"Precision: "
        f"{precision_score(y_test, y_pred):.4f}"
    )

    print(
        f"Recall:    "
        f"{recall_score(y_test, y_pred):.4f}"
    )

    print(
        f"F1 Score:  "
        f"{f1_score(y_test, y_pred):.4f}"
    )

    print(
        f"ROC-AUC:   "
        f"{roc_auc_score(y_test, y_probability):.4f}"
    )

    print(
        f"PR-AUC:    "
        f"{average_precision_score(y_test, y_probability):.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

def evaluate_random_forest(model, X_test, y_test):
    """Evaluate the Random Forest model."""

    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(X_test)[:, 1]

    print("\n========== RANDOM FOREST EVALUATION ==========")

    print(
        f"Accuracy:  "
        f"{accuracy_score(y_test, y_pred):.4f}"
    )

    print(
        f"Precision: "
        f"{precision_score(y_test, y_pred):.4f}"
    )

    print(
        f"Recall:    "
        f"{recall_score(y_test, y_pred):.4f}"
    )

    print(
        f"F1 Score:  "
        f"{f1_score(y_test, y_pred):.4f}"
    )

    print(
        f"ROC-AUC:   "
        f"{roc_auc_score(y_test, y_probability):.4f}"
    )

    print(
        f"PR-AUC:    "
        f"{average_precision_score(y_test, y_probability):.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

def evaluate_xgboost(model, X_test, y_test):
    """Evaluate the XGBoost model."""

    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(X_test)[:, 1]

    print("\n========== XGBOOST EVALUATION ==========")

    print(
        f"Accuracy:  "
        f"{accuracy_score(y_test, y_pred):.4f}"
    )

    print(
        f"Precision: "
        f"{precision_score(y_test, y_pred):.4f}"
    )

    print(
        f"Recall:    "
        f"{recall_score(y_test, y_pred):.4f}"
    )

    print(
        f"F1 Score:  "
        f"{f1_score(y_test, y_pred):.4f}"
    )

    print(
        f"ROC-AUC:   "
        f"{roc_auc_score(y_test, y_probability):.4f}"
    )

    print(
        f"PR-AUC:    "
        f"{average_precision_score(y_test, y_probability):.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

def main():

    print("Loading ML datasets...")

    X_train, X_test, y_train, y_test = load_data()

    print(
        f"Training data: {X_train.shape}"
    )

    print(
        f"Testing data:  {X_test.shape}"
    )

    print("\nTraining Logistic Regression...")

    model, scaler = train_model(
        X_train,
        y_train
    )

    print("Training completed!")

    evaluate_model(
        model,
        scaler,
        X_test,
        y_test
    )

    print("\nTraining Decision Tree...")

    tree_model = train_decision_tree(
        X_train,
        y_train
    )

    print("Decision Tree training completed!")

    evaluate_tree(
        tree_model,
        X_test,
        y_test
    )

    print("\nTraining Random Forest...")

    random_forest_model = train_random_forest(
        X_train,
        y_train
    )

    print("Random Forest training completed!")

    evaluate_random_forest(
        random_forest_model,
        X_test,
        y_test
    )

    print("\nTraining XGBoost...")

    xgb_model = train_xgboost(
        X_train,
        y_train
    )

    print("XGBoost training completed!")

    evaluate_xgboost(
        xgb_model,
        X_test,
        y_test
    )

if __name__ == "__main__":
    main()