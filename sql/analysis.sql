-- 1. Basic customer overview

SELECT
    plan_type,
    COUNT(*) AS customer_count,
    ROUND(AVG(age), 2) AS average_age
FROM customers
GROUP BY plan_type
ORDER BY customer_count DESC;

-- 2. Total revenue per customer

SELECT
    c.customer_id,
    c.city,
    c.plan_type,
    COUNT(o.order_id) AS total_orders,
    ROUND(SUM(o.amount), 2) AS total_revenue
FROM customers c
JOIN orders o
    ON c.customer_id = o.customer_id
GROUP BY
    c.customer_id,
    c.city,
    c.plan_type
ORDER BY total_revenue DESC;

-- 3. Average order value per customer

SELECT
    c.customer_id,
    c.plan_type,
    COUNT(o.order_id) AS total_orders,
    ROUND(SUM(o.amount), 2) AS total_revenue,
    ROUND(AVG(o.amount), 2) AS average_order_value
FROM customers c
JOIN orders o
    ON c.customer_id = o.customer_id
GROUP BY
    c.customer_id,
    c.plan_type
ORDER BY average_order_value DESC;

-- 4. Payment failure analysis

SELECT
    c.customer_id,
    c.plan_type,
    COUNT(p.payment_id) AS total_payments,
    SUM(
        CASE
            WHEN p.payment_status = 'Failed' THEN 1
            ELSE 0
        END
    ) AS failed_payments,
    ROUND(
        100.0 * SUM(
            CASE
                WHEN p.payment_status = 'Failed' THEN 1
                ELSE 0
            END
        ) / COUNT(p.payment_id),
        2
    ) AS failure_rate
FROM customers c
JOIN payments p
    ON c.customer_id = p.customer_id
GROUP BY
    c.customer_id,
    c.plan_type
ORDER BY failure_rate DESC;

-- 5. Support ticket analysis

SELECT
    c.customer_id,
    c.plan_type,
    COUNT(t.ticket_id) AS total_tickets,
    ROUND(AVG(t.resolution_time), 2) AS avg_resolution_time,
    ROUND(AVG(t.satisfaction_score), 2) AS avg_satisfaction_score
FROM customers c
JOIN support_tickets t
    ON c.customer_id = t.customer_id
GROUP BY
    c.customer_id,
    c.plan_type
ORDER BY avg_satisfaction_score ASC;

-- 6. Customer order recency

SELECT
    c.customer_id,
    c.plan_type,
    MAX(o.order_date) AS last_order_date,
    CURRENT_DATE - MAX(o.order_date) AS days_since_last_order
FROM customers c
JOIN orders o
    ON c.customer_id = o.customer_id
GROUP BY
    c.customer_id,
    c.plan_type
ORDER BY days_since_last_order DESC;

-- 7. Customer-level feature dataset

WITH order_features AS (
    SELECT
        customer_id,
        COUNT(order_id) AS total_orders,
        ROUND(SUM(amount), 2) AS total_revenue,
        ROUND(AVG(amount), 2) AS average_order_value,
        MAX(order_date) AS last_order_date
    FROM orders
    GROUP BY customer_id
),

payment_features AS (
    SELECT
        customer_id,
        COUNT(payment_id) AS total_payments,
        SUM(
            CASE
                WHEN payment_status = 'Failed' THEN 1
                ELSE 0
            END
        ) AS failed_payments
    FROM payments
    GROUP BY customer_id
),

support_features AS (
    SELECT
        customer_id,
        COUNT(ticket_id) AS total_tickets,
        ROUND(AVG(resolution_time), 2) AS avg_resolution_time,
        ROUND(AVG(satisfaction_score), 2) AS avg_satisfaction_score
    FROM support_tickets
    GROUP BY customer_id
)

SELECT
    c.customer_id,
    c.age,
    c.city,
    c.plan_type,

    -- Order features
    COALESCE(o.total_orders, 0) AS total_orders,
    COALESCE(o.total_revenue, 0) AS total_revenue,
    COALESCE(o.average_order_value, 0) AS average_order_value,
    o.last_order_date,

    -- Payment features
    COALESCE(p.total_payments, 0) AS total_payments,
    COALESCE(p.failed_payments, 0) AS failed_payments,

    CASE
        WHEN p.total_payments > 0
        THEN ROUND(
            100.0 * p.failed_payments / p.total_payments,
            2
        )
        ELSE 0
    END AS payment_failure_rate,

    -- Support features
    COALESCE(s.total_tickets, 0) AS total_tickets,
    COALESCE(s.avg_resolution_time, 0) AS avg_resolution_time,
    COALESCE(s.avg_satisfaction_score, 0) AS avg_satisfaction_score

FROM customers c

LEFT JOIN order_features o
    ON c.customer_id = o.customer_id

LEFT JOIN payment_features p
    ON c.customer_id = p.customer_id

LEFT JOIN support_features s
    ON c.customer_id = s.customer_id

ORDER BY c.customer_id;