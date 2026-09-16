import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

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


def tune_random_forest(X_train, y_train):
    """Find the best Random Forest hyperparameters."""

    model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [5, 8, 12],
        "min_samples_split": [10, 20],
        "min_samples_leaf": [5, 10]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    print("Starting Random Forest tuning...\n")

    grid_search.fit(X_train, y_train)

    print("\n" + "=" * 60)
    print("RANDOM FOREST TUNING RESULTS")
    print("=" * 60)

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print(
        f"\nBest CV PR-AUC: "
        f"{grid_search.best_score_:.4f}"
    )

    return grid_search


def main():

    X_train, y_train = load_training_data()

    print("Training data:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    grid_search = tune_random_forest(
        X_train,
        y_train
    )

    print("\nBest model:")
    print(grid_search.best_estimator_)


if __name__ == "__main__":
    main()