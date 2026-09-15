import pandas as pd

from src.config import RAW_DATA_DIR
from src.database.connection import engine


TABLES = {
    "customers.csv": "customers",
    "orders.csv": "orders",
    "payments.csv": "payments",
    "subscriptions.csv": "subscriptions",
    "support_tickets.csv": "support_tickets",
}


def load_csv_to_database(file_name, table_name):
    """Load a CSV file into a PostgreSQL table."""

    file_path = RAW_DATA_DIR / file_name

    print(f"Loading {file_name}...")

    df = pd.read_csv(file_path)

    df.to_sql(
        table_name,
        engine,
        if_exists="append",
        index=False,
        method="multi",
    )

    print(f"Loaded {len(df)} rows into {table_name}")


def main():

    print("Starting database loading...\n")

    for file_name, table_name in TABLES.items():

        load_csv_to_database(
            file_name,
            table_name
        )

    print("\nAll data loaded successfully!")


if __name__ == "__main__":
    main()