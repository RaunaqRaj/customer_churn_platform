import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from src.config import PROCESSED_DATA_DIR, MODEL_DIR


def load_training_data():
    """Load the training data."""

    X_train = pd.read_csv(
        PROCESSED_DATA_DIR / "X_train.csv"
    )

    y_train = pd.read_csv(
        PROCESSED_DATA_DIR / "y_train.csv"
    ).squeeze()

    return X_train, y_train


def train_final_model(X_train, y_train):
    """Train the tuned Random Forest model."""

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


def main():

    print("=" * 60)
    print("TRAINING FINAL CHURN MODEL")
    print("=" * 60)

    X_train, y_train = load_training_data()

    print("\nTraining data:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    model = train_final_model(
        X_train,
        y_train
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = MODEL_DIR / "churn_model.pkl"

    joblib.dump(
        model,
        model_path
    )

    print("\nModel trained successfully!")

    print("\nModel saved to:")
    print(model_path)


if __name__ == "__main__":
    main()