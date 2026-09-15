import pandas as pd


# ==========================================
# LOAD DATA
# ==========================================

customers = pd.read_csv("data/raw/customers.csv")
orders = pd.read_csv("data/raw/orders.csv")
payments = pd.read_csv("data/raw/payments.csv")
subscriptions = pd.read_csv("data/raw/subscriptions.csv")
support_tickets = pd.read_csv("data/raw/support_tickets.csv")


# ==========================================
# 1. CHECK CUSTOMER IDs
# ==========================================

print("\n===== CUSTOMER ID CHECK =====")

duplicate_customers = customers["customer_id"].duplicated().sum()

print("Duplicate customer IDs:", duplicate_customers)


# ==========================================
# 2. CREATE VALID CUSTOMER ID SET
# ==========================================

customer_ids = set(customers["customer_id"])


# ==========================================
# 3. VALIDATE ORDERS
# ==========================================

invalid_orders = ~orders["customer_id"].isin(customer_ids)

print("\n===== ORDERS CHECK =====")
print("Total orders:", len(orders))
print("Orders with invalid customer ID:", invalid_orders.sum())


# ==========================================
# 4. VALIDATE PAYMENTS
# ==========================================

invalid_payments = ~payments["customer_id"].isin(customer_ids)

print("\n===== PAYMENTS CHECK =====")
print("Total payments:", len(payments))
print("Payments with invalid customer ID:", invalid_payments.sum())


# ==========================================
# 5. VALIDATE SUBSCRIPTIONS
# ==========================================

invalid_subscriptions = ~subscriptions["customer_id"].isin(customer_ids)

print("\n===== SUBSCRIPTIONS CHECK =====")
print("Total subscriptions:", len(subscriptions))
print(
    "Subscriptions with invalid customer ID:",
    invalid_subscriptions.sum()
)


# ==========================================
# 6. VALIDATE SUPPORT TICKETS
# ==========================================

invalid_tickets = ~support_tickets["customer_id"].isin(customer_ids)

print("\n===== SUPPORT TICKETS CHECK =====")
print("Total tickets:", len(support_tickets))
print("Tickets with invalid customer ID:", invalid_tickets.sum())


# ==========================================
# FINAL RESULT
# ==========================================

print("\n===================================")
print("DATA VALIDATION COMPLETED")
print("===================================")