import pandas as pd

customers = pd.read_csv("data/raw/customers.csv")
orders = pd.read_csv("data/raw/orders.csv")
payments = pd.read_csv("data/raw/payments.csv")
subscriptions = pd.read_csv("data/raw/subscriptions.csv")
support_tickets = pd.read_csv("data/raw/support_tickets.csv")

print("\n===== CUSTOMERS =====")
print(customers.head())
print(customers.shape)

print("\n===== ORDERS =====")
print(orders.head())
print(orders.shape)

print("\n===== PAYMENTS =====")
print(payments.head())
print(payments.shape)

print("\n===== SUBSCRIPTIONS =====")
print(subscriptions.head())
print(subscriptions.shape)

print("\n===== SUPPORT TICKETS =====")
print(support_tickets.head())
print(support_tickets.shape)