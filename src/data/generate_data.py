import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.config import RAW_DATA_DIR


random.seed(42)
np.random.seed(42)


def random_date(start_date, end_date):
    days = (end_date - start_date).days
    random_days = random.randint(0, days)

    return start_date + timedelta(days=random_days)


def generate_customers(n=5000):

    cities = [
        "Delhi",
        "Mumbai",
        "Bangalore",
        "Hyderabad",
        "Chennai",
        "Pune",
        "Kolkata",
        "Jaipur",
    ]

    plans = [
        "Basic",
        "Standard",
        "Premium",
    ]

    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 12, 31)

    customers = []

    for i in range(1, n + 1):

        signup_date = random_date(start_date, end_date)

        customers.append({
            "customer_id": f"C{i:05d}",
            "age": random.randint(18, 65),
            "city": random.choice(cities),
            "signup_date": signup_date,
            "plan_type": random.choice(plans),
        })

    return pd.DataFrame(customers)


def main():

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    customers = generate_customers()

    customers.to_csv(
        RAW_DATA_DIR / "customers.csv",
        index=False
    )

    print("Customer data generated successfully.")
    print(f"Rows: {len(customers)}")


if __name__ == "__main__":
    main()