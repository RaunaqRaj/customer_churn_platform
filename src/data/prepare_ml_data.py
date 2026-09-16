import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DATA_DIR


def load_dataset():
    """Load the processed customer ML dataset."""
    file_path = PROCESSED_DATA_DIR / "customer_ml_dataset.csv"

    df = pd.read_csv(file_path)

    return df


def prepare_features(df):
    """Prepare feature matrix X and target variable y."""

    # Target variable
    y = df["churn"]

    # Columns that should NOT be used as model features
    columns_to_drop = [
        "customer_id",
        "signup_date",
        "last_order_date",
        "churn"
    ]

    X = df.drop(columns=columns_to_drop)

    # Convert categorical columns into numeric dummy variables
    X = pd.get_dummies(
        X,
        columns=["city", "plan_type"],
        drop_first=True
    )

    return X, y


def main():

    print("Loading processed dataset...\n")

    df = load_dataset()

    print("Original dataset shape:")
    print(df.shape)

    # Prepare features and target
    X, y = prepare_features(df)

    print("\nFeature matrix shape:")
    print(X.shape)

    print("\nTarget shape:")
    print(y.shape)

    print("\nFeatures:")
    print(X.columns.tolist())

    print("\nTarget distribution:")
    print(y.value_counts())

    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTrain/Test split:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    # Save datasets
    X_train.to_csv(
        PROCESSED_DATA_DIR / "X_train.csv",
        index=False
    )

    X_test.to_csv(
        PROCESSED_DATA_DIR / "X_test.csv",
        index=False
    )

    y_train.to_csv(
        PROCESSED_DATA_DIR / "y_train.csv",
        index=False
    )

    y_test.to_csv(
        PROCESSED_DATA_DIR / "y_test.csv",
        index=False
    )

    print("\nML datasets saved successfully!")
    print("\nFeature data types:")
    print(X.dtypes)

    print("\nMissing values in X:")
    print(X.isnull().sum())

    print("\nUnique target values:")
    print(y.unique())

if __name__ == "__main__":
    main()