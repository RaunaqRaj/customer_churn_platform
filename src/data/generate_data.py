import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.config import RAW_DATA_DIR


# ---------------------------------------------------------
# RANDOM SEED
# ---------------------------------------------------------

random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------
# HELPER FUNCTION
# ---------------------------------------------------------

def random_date(start_date, end_date):
    """Generate a random date between two dates."""

    days = (end_date - start_date).days

    random_days = random.randint(0, days)

    return start_date + timedelta(days=random_days)


# ---------------------------------------------------------
# 1. CUSTOMERS
# ---------------------------------------------------------

def generate_customers(n=5000):

    cities = [
        "Delhi",
        "Mumbai",
        "Bangalore",
        "Hyderabad",
        "Chennai",
        "Pune",
        "Kolkata",
        "Jaipur"
    ]

    plans = [
        "Basic",
        "Standard",
        "Premium"
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
            "plan_type": random.choice(plans)
        })

    return pd.DataFrame(customers)


# ---------------------------------------------------------
# 2. ORDERS
# ---------------------------------------------------------

def generate_orders(customers):

    orders = []

    order_id = 1

    for _, customer in customers.iterrows():

        customer_id = customer["customer_id"]

        signup_date = pd.to_datetime(customer["signup_date"])

        # Number of orders for this customer
        number_of_orders = np.random.poisson(12)

        for _ in range(number_of_orders):

            order_date = signup_date + timedelta(
                days=random.randint(1, 900)
            )

            # Don't create future orders
            if order_date > datetime(2025, 12, 31):
                continue

            amount = round(
                random.uniform(300, 10000),
                2
            )

            orders.append({
                "order_id": f"O{order_id:07d}",
                "customer_id": customer_id,
                "order_date": order_date,
                "amount": amount,
                "product_id": f"P{random.randint(1, 100):03d}"
            })

            order_id += 1

    return pd.DataFrame(orders)


# ---------------------------------------------------------
# 3. PAYMENTS
# ---------------------------------------------------------

def generate_payments(customers):

    payments = []

    payment_id = 1

    payment_statuses = [
        "Success",
        "Success",
        "Success",
        "Success",
        "Failed"
    ]

    for _, customer in customers.iterrows():

        customer_id = customer["customer_id"]

        signup_date = pd.to_datetime(customer["signup_date"])

        number_of_payments = random.randint(3, 18)

        for _ in range(number_of_payments):

            payment_date = signup_date + timedelta(
                days=random.randint(1, 900)
            )

            if payment_date > datetime(2025, 12, 31):
                continue

            amount = round(
                random.uniform(500, 5000),
                2
            )

            status = random.choice(payment_statuses)

            payments.append({
                "payment_id": f"PAY{payment_id:07d}",
                "customer_id": customer_id,
                "payment_date": payment_date,
                "amount": amount,
                "payment_status": status
            })

            payment_id += 1

    return pd.DataFrame(payments)


# ---------------------------------------------------------
# 4. SUBSCRIPTIONS
# ---------------------------------------------------------

def generate_subscriptions(customers):

    subscriptions = []

    subscription_id = 1

    monthly_fees = {
        "Basic": 499,
        "Standard": 999,
        "Premium": 1999
    }

    for _, customer in customers.iterrows():

        customer_id = customer["customer_id"]

        signup_date = pd.to_datetime(customer["signup_date"])

        plan = customer["plan_type"]

        monthly_fee = monthly_fees[plan]

        subscriptions.append({
            "subscription_id": f"S{subscription_id:07d}",
            "customer_id": customer_id,
            "start_date": signup_date,
            "end_date": datetime(2025, 12, 31),
            "plan": plan,
            "monthly_fee": monthly_fee
        })

        subscription_id += 1

    return pd.DataFrame(subscriptions)


# ---------------------------------------------------------
# 5. SUPPORT TICKETS
# ---------------------------------------------------------

def generate_support_tickets(customers):

    tickets = []

    ticket_id = 1

    categories = [
        "Payment Issue",
        "Technical Issue",
        "Delivery",
        "Account",
        "Product"
    ]

    for _, customer in customers.iterrows():

        customer_id = customer["customer_id"]

        signup_date = pd.to_datetime(customer["signup_date"])

        number_of_tickets = np.random.poisson(3)

        for _ in range(number_of_tickets):

            created_at = signup_date + timedelta(
                days=random.randint(1, 900)
            )

            if created_at > datetime(2025, 12, 31):
                continue

            tickets.append({
                "ticket_id": f"T{ticket_id:07d}",
                "customer_id": customer_id,
                "created_at": created_at,
                "category": random.choice(categories),
                "resolution_time": random.randint(1, 72),
                "satisfaction_score": random.randint(1, 5)
            })

            ticket_id += 1

    return pd.DataFrame(tickets)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Generating customers...")

    customers = generate_customers(5000)

    print("Generating orders...")

    orders = generate_orders(customers)

    print("Generating payments...")

    payments = generate_payments(customers)

    print("Generating subscriptions...")

    subscriptions = generate_subscriptions(customers)

    print("Generating support tickets...")

    support_tickets = generate_support_tickets(customers)


    # -----------------------------------------------------
    # SAVE DATA
    # -----------------------------------------------------

    customers.to_csv(
        RAW_DATA_DIR / "customers.csv",
        index=False
    )

    orders.to_csv(
        RAW_DATA_DIR / "orders.csv",
        index=False
    )

    payments.to_csv(
        RAW_DATA_DIR / "payments.csv",
        index=False
    )

    subscriptions.to_csv(
        RAW_DATA_DIR / "subscriptions.csv",
        index=False
    )

    support_tickets.to_csv(
        RAW_DATA_DIR / "support_tickets.csv",
        index=False
    )


    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    print("\nData generation completed successfully!")

    print(f"Customers: {len(customers)}")
    print(f"Orders: {len(orders)}")
    print(f"Payments: {len(payments)}")
    print(f"Subscriptions: {len(subscriptions)}")
    print(f"Support Tickets: {len(support_tickets)}")


if __name__ == "__main__":
    main()