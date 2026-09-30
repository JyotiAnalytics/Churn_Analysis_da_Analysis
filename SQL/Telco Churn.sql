use churn;

DROP TABLE IF EXISTS [dbo].[WA_Fn-UseC_-Telco-Customer-Churn];

EXEC sp_rename 'WA_Fn-UseC_-Telco-Customer-Churn', 'Telco_Customer_Churn';


SELECT *
FROM Telco_Customer_Churn;


-- TELCO CUSTOMER CHURN SQL ANALYSIS
-- Load the CSV into a table named customers before running.

-- TELCO CUSTOMER CHURN SQL ANALYSIS (SQL Server version)

-- 1. Total customers
SELECT COUNT(DISTINCT customerID) AS total_customers
FROM Telco_Customer_Churn;

-- 2. Overall churn rate
SELECT
    COUNT(DISTINCT customerID) AS total_customers,
    SUM(CASE WHEN CAST(Churn AS VARCHAR(3)) IN ('Yes','1') THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(
        100.0 * AVG(CASE WHEN CAST(Churn AS VARCHAR(3)) IN ('Yes','1') THEN 1.0 ELSE 0.0 END), 2
    ) AS churn_rate
FROM Telco_Customer_Churn;

-- 3. Churn by contract type
SELECT
    Contract,
    COUNT(*) AS customers,
    SUM(CASE WHEN CAST(Churn AS VARCHAR(3)) IN ('Yes','1') THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(
        100.0 * AVG(CASE WHEN CAST(Churn AS VARCHAR(3)) IN ('Yes','1') THEN 1.0 ELSE 0.0 END), 2
    ) AS churn_rate,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges,
    ROUND(SUM(MonthlyCharges), 2) AS monthly_revenue
FROM Telco_Customer_Churn
GROUP BY Contract
ORDER BY churn_rate DESC;

-- 4. Churn by payment method
SELECT
    PaymentMethod,
    COUNT(*) AS customers,
    SUM(CASE WHEN CAST(Churn AS VARCHAR(3)) IN ('Yes','1') THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(
        100.0 * AVG(CASE WHEN CAST(Churn AS VARCHAR(3)) IN ('Yes','1') THEN 1.0 ELSE 0.0 END), 2
    ) AS churn_rate
FROM Telco_Customer_Churn
GROUP BY PaymentMethod
ORDER BY churn_rate DESC;

-- 5. Churn by internet service
SELECT
    InternetService,
    COUNT(*) AS customers,
    ROUND(
        100.0 * AVG(CASE WHEN CAST(Churn AS VARCHAR(3)) IN ('Yes','1') THEN 1.0 ELSE 0.0 END), 2
    ) AS churn_rate
FROM Telco_Customer_Churn
GROUP BY InternetService
ORDER BY churn_rate DESC;

-- 6. Churn by contract + payment method
SELECT
    Contract,
    PaymentMethod,
    COUNT(*) AS customers,
    ROUND(
        100.0 * AVG(CASE WHEN CAST(Churn AS VARCHAR(3)) IN ('Yes','1') THEN 1.0 ELSE 0.0 END), 2
    ) AS churn_rate
FROM Telco_Customer_Churn
GROUP BY Contract, PaymentMethod
ORDER BY churn_rate DESC;

-- 7. Risk ranking of segments
WITH ranked_segments AS (
    SELECT
        Contract,
        PaymentMethod,
        COUNT(*) AS customers,
        AVG(CASE WHEN CAST(Churn AS VARCHAR(3)) IN ('Yes','1') THEN 1.0 ELSE 0.0 END) AS churn_rate
    FROM Telco_Customer_Churn
    GROUP BY Contract, PaymentMethod
)
SELECT
    *,
    RANK() OVER (ORDER BY churn_rate DESC) AS risk_rank
FROM ranked_segments
ORDER BY risk_rank;

-- 8. Top 100 churned customers by monthly charges
SELECT TOP 100
    customerID,
    Contract,
    tenure,
    MonthlyCharges,
    TRY_CAST(TotalCharges AS DECIMAL(10,2)) AS TotalCharges,
    Churn
FROM Telco_Customer_Churn
WHERE CAST(Churn AS VARCHAR(3)) IN ('Yes','1')
ORDER BY MonthlyCharges DESC;
