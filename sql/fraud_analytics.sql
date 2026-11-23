
-- =====================================================
-- TxnSentry: Financial Fraud Detection SQL Analytics
-- Database table: transactions
-- Class: 0 = Legitimate, 1 = Fraud
-- =====================================================


-- 1. Overall transaction summary

SELECT
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END)
        AS fraud_transactions,
    SUM(CASE WHEN Class = 0 THEN 1 ELSE 0 END)
        AS legitimate_transactions,
    ROUND(
        100.0 * SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END)
        / COUNT(*), 4
    ) AS fraud_percentage
FROM transactions;


-- 2. Transaction amount statistics by class

SELECT
    Class,
    COUNT(*) AS transaction_count,
    ROUND(AVG(Amount), 2) AS average_amount,
    ROUND(MIN(Amount), 2) AS minimum_amount,
    ROUND(MAX(Amount), 2) AS maximum_amount
FROM transactions
GROUP BY Class
ORDER BY Class;


-- 3. Fraud distribution by transaction amount

SELECT
    CASE
        WHEN Amount < 50 THEN 'Below 50'
        WHEN Amount < 100 THEN '50-99.99'
        WHEN Amount < 500 THEN '100-499.99'
        WHEN Amount < 1000 THEN '500-999.99'
        ELSE '1000 and above'
    END AS amount_band,

    COUNT(*) AS total_transactions,

    SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END)
        AS fraud_transactions,

    ROUND(
        100.0 * SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END)
        / COUNT(*), 4
    ) AS fraud_percentage

FROM transactions
GROUP BY amount_band
ORDER BY
    CASE amount_band
        WHEN 'Below 50' THEN 1
        WHEN '50-99.99' THEN 2
        WHEN '100-499.99' THEN 3
        WHEN '500-999.99' THEN 4
        ELSE 5
    END;


-- 4. Fraud and legitimate transaction counts by time bucket
-- Time is elapsed seconds, not a real calendar timestamp.

SELECT
    CAST((Time % 86400) / 3600 AS INTEGER) AS hour_bucket,
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END)
        AS fraud_transactions
FROM transactions
GROUP BY hour_bucket
ORDER BY hour_bucket;


-- 5. Most influential anonymized features:
-- Average values by historical class label.

SELECT
    Class,
    ROUND(AVG(V14), 5) AS avg_V14,
    ROUND(AVG(V12), 5) AS avg_V12,
    ROUND(AVG(V10), 5) AS avg_V10,
    ROUND(AVG(V4), 5) AS avg_V4,
    ROUND(AVG(V17), 5) AS avg_V17
FROM transactions
GROUP BY Class
ORDER BY Class;


-- 6. Highest-value fraud transactions

SELECT
    Time,
    Amount,
    V14,
    V12,
    V10,
    V4,
    V17
FROM transactions
WHERE Class = 1
ORDER BY Amount DESC
LIMIT 20;


-- 7. Fraud rate by transaction amount band
-- Only bands with at least 100 transactions are included.

SELECT
    CASE
        WHEN Amount < 50 THEN 'Below 50'
        WHEN Amount < 100 THEN '50-99.99'
        WHEN Amount < 500 THEN '100-499.99'
        WHEN Amount < 1000 THEN '500-999.99'
        ELSE '1000 and above'
    END AS amount_band,

    COUNT(*) AS total_transactions,

    SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END)
        AS fraud_count,

    ROUND(
        100.0 * SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END)
        / COUNT(*), 4
    ) AS fraud_rate_percent

FROM transactions
GROUP BY amount_band
HAVING COUNT(*) >= 100
ORDER BY fraud_rate_percent DESC;
