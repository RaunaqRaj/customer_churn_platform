-- ============================================================
-- CUSTOMER CHURN LABEL & ML FEATURE DATASET
-- Snapshot Date: 2025-09-30
-- Churn Window: 90 days
-- ============================================================


-- 1. Check the available order date range

SELECT
    MIN(order_date) AS first_order_date,
    MAX(order_date) AS last_order_date
FROM orders;


-- 2. Create churn labels
--
-- A customer is considered churned if they:
--   1. Were already registered by the snapshot date
--   2. Had NO order during the following 90 days
--
-- Snapshot Date: 2025-09-30
-- Prediction Window: 2025-10-01 to 2025-12-29

SELECT
    c.customer_id,
    CASE
        WHEN COUNT(o.order_id) = 0 THEN 1
        ELSE 0
    END AS churn
FROM customers c
LEFT JOIN orders o
    ON c.customer_id = o.customer_id
    AND o.order_date > DATE '2025-09-30'
    AND o.order_date <= DATE '2025-12-29'
WHERE c.signup_date <= DATE '2025-09-30'
GROUP BY c.customer_id
ORDER BY c.customer_id;


-- 3. Check the churn distribution

SELECT
    churn,
    COUNT(*) AS customer_count,
    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS percentage
FROM (
    SELECT
        c.customer_id,
        CASE
            WHEN COUNT(o.order_id) = 0 THEN 1
            ELSE 0
        END AS churn
    FROM customers c
    LEFT JOIN orders o
        ON c.customer_id = o.customer_id
        AND o.order_date > DATE '2025-09-30'
        AND o.order_date <= DATE '2025-12-29'
    WHERE c.signup_date <= DATE '2025-09-30'
    GROUP BY c.customer_id
) churn_data
GROUP BY churn
ORDER BY churn;


-- 4. Build ML feature dataset

CREATE OR REPLACE VIEW customer_ml_features AS

WITH eligible_customers AS (

    SELECT
        customer_id,
        age,
        city,
        plan_type,
        signup_date

    FROM customers

    WHERE signup_date <= DATE '2025-09-30'
),


-- ============================================================
-- ORDER FEATURES
-- ============================================================

order_features AS (

    SELECT
        customer_id,

        COUNT(order_id) AS total_orders,

        ROUND(SUM(amount), 2) AS total_revenue,

        ROUND(AVG(amount), 2) AS average_order_value,

        MAX(order_date) AS last_order_date

    FROM orders

    WHERE order_date <= DATE '2025-09-30'

    GROUP BY customer_id
),


-- ============================================================
-- PAYMENT FEATURES
-- ============================================================

payment_features AS (

    SELECT
        customer_id,

        COUNT(payment_id) AS total_payments,

        SUM(
            CASE
                WHEN payment_status = 'Failed'
                THEN 1
                ELSE 0
            END
        ) AS failed_payments

    FROM payments

    WHERE payment_date <= DATE '2025-09-30'

    GROUP BY customer_id
),


-- ============================================================
-- SUPPORT FEATURES
-- ============================================================

support_features AS (

    SELECT
        customer_id,

        COUNT(ticket_id) AS total_tickets,

        ROUND(
            AVG(resolution_time),
            2
        ) AS avg_resolution_time,

        ROUND(
            AVG(satisfaction_score),
            2
        ) AS avg_satisfaction_score

    FROM support_tickets

    WHERE created_at <= DATE '2025-09-30'

    GROUP BY customer_id
),


-- ============================================================
-- CHURN LABEL
-- ============================================================

churn_labels AS (

    SELECT
        c.customer_id,

        CASE
            WHEN COUNT(o.order_id) = 0
            THEN 1
            ELSE 0
        END AS churn

    FROM eligible_customers c

    LEFT JOIN orders o
        ON c.customer_id = o.customer_id

        AND o.order_date > DATE '2025-09-30'

        AND o.order_date <= DATE '2025-12-29'

    GROUP BY c.customer_id
)


-- ============================================================
-- FINAL ML DATASET
-- ============================================================

SELECT

    c.customer_id,

    c.age,

    c.city,

    c.plan_type,

    c.signup_date,


    -- Customer tenure

    DATE '2025-09-30' - c.signup_date
        AS tenure_days,


    -- Order features

    COALESCE(
        o.total_orders,
        0
    ) AS total_orders,

    COALESCE(
        o.total_revenue,
        0
    ) AS total_revenue,

    COALESCE(
        o.average_order_value,
        0
    ) AS average_order_value,


    o.last_order_date,


    -- Recency

    CASE

        WHEN o.last_order_date IS NOT NULL

        THEN DATE '2025-09-30'
             - o.last_order_date

        ELSE NULL

    END AS days_since_last_order,


    -- Payment features

    COALESCE(
        p.total_payments,
        0
    ) AS total_payments,

    COALESCE(
        p.failed_payments,
        0
    ) AS failed_payments,


    CASE

        WHEN p.total_payments > 0

        THEN ROUND(
            100.0
            * p.failed_payments
            / p.total_payments,
            2
        )

        ELSE 0

    END AS payment_failure_rate,


    -- Support features

    COALESCE(
        s.total_tickets,
        0
    ) AS total_tickets,

    COALESCE(
        s.avg_resolution_time,
        0
    ) AS avg_resolution_time,

    COALESCE(
        s.avg_satisfaction_score,
        0
    ) AS avg_satisfaction_score,


    -- Target

    ch.churn


FROM eligible_customers c


LEFT JOIN order_features o

    ON c.customer_id = o.customer_id


LEFT JOIN payment_features p

    ON c.customer_id = p.customer_id


LEFT JOIN support_features s

    ON c.customer_id = s.customer_id


LEFT JOIN churn_labels ch

    ON c.customer_id = ch.customer_id


ORDER BY c.customer_id;


-- 5. Preview the ML feature dataset

SELECT *
FROM customer_ml_features
LIMIT 10;