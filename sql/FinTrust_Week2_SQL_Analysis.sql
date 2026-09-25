-- ==============================================================================
-- Project: FinTrust Digital Bank — Week 2 Analysis
-- Deliverable: FinTrust_Week2_SQL_Analysis.sql
-- Compatible: PostgreSQL, MySQL, DuckDB, SQLite
-- ==============================================================================

-- 1. Customer Segment Commercial Performance & Volume Mix
-- Business Interpretation: "Everyday" accounts represent the bulk of transaction volume 
-- (5,644 txs, 261.5M NGN). "SME" accounts exhibit the highest average transaction size (49,115 NGN).
SELECT 
    c.Customer_Segment,
    COUNT(t.Transaction_ID) AS Total_Transactions,
    ROUND(SUM(t.Amount_NGN), 2) AS Total_Value_NGN,
    ROUND(AVG(t.Amount_NGN), 2) AS Avg_Ticket_NGN
FROM FinTrust_Customer_Data c
JOIN FinTrust_Transaction_Data t ON c.Customer_ID = t.Customer_ID
GROUP BY c.Customer_Segment
ORDER BY Total_Value_NGN DESC;

-- 2. Channel Success & Failure Rates
-- Business Interpretation: Mobile App drives 42.5% of overall bank volume (5,102 txs) 
-- but experiences the highest failure rate (5.80%), identifying key digital infrastructure friction.
SELECT 
    Channel,
    COUNT(*) AS Total_Attempts,
    SUM(CASE WHEN Transaction_Status = 'Successful' THEN 1 ELSE 0 END) AS Success_Count,
    ROUND(SUM(CASE WHEN Transaction_Status = 'Successful' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS Success_Rate_Pct,
    ROUND(SUM(CASE WHEN Transaction_Status = 'Failed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS Failure_Rate_Pct
FROM FinTrust_Transaction_Data
GROUP BY Channel
ORDER BY Total_Attempts DESC;

-- 3. Synthetic Risk-Review Flag Rate by Transaction Modality
-- Business Interpretation: Bill Payments and Transfers trigger the highest proportion of 
-- risk flags (~20.5%), warranting review of automated routing rules.
SELECT 
    Transaction_Type,
    COUNT(*) AS Total_Volume,
    SUM(CASE WHEN Risk_Review_Flag = 'Yes' THEN 1 ELSE 0 END) AS Flagged_Count,
    ROUND(SUM(CASE WHEN Risk_Review_Flag = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS Flag_Rate_Pct
FROM FinTrust_Transaction_Data
GROUP BY Transaction_Type
ORDER BY Flag_Rate_Pct DESC;

-- 4. Cross-Border vs. Domestic Operational Friction
-- Business Interpretation: Cross-border transactions account for 4.0% of activity (480 txs), 
-- but trigger review flags at nearly double the rate of domestic operations (36.88% vs 18.88%).
SELECT 
    International_Transaction,
    COUNT(*) AS Transaction_Count,
    ROUND(SUM(Amount_NGN), 2) AS Total_Value_NGN,
    ROUND(AVG(Amount_NGN), 2) AS Avg_Value_NGN,
    ROUND(SUM(CASE WHEN Transaction_Status = 'Failed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS Failure_Rate_Pct,
    ROUND(SUM(CASE WHEN Risk_Review_Flag = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS Risk_Flag_Rate_Pct
FROM FinTrust_Transaction_Data
GROUP BY International_Transaction;

-- 5. Digital Engagement Score Impact on Customer Frequency
-- Business Interpretation: Higher digital adoption directly increases customer activity: 
-- users in the top tier (score >= 80) average 8.41 transactions vs 7.62 for low-engagement users.
SELECT 
    CASE 
        WHEN c.Digital_Engagement_Score >= 80 THEN 'High (80-100)'
        WHEN c.Digital_Engagement_Score >= 50 THEN 'Medium (50-79)'
        ELSE 'Low (<50)'
    END AS Engagement_Tier,
    COUNT(DISTINCT c.Customer_ID) AS Total_Customers,
    COUNT(t.Transaction_ID) AS Total_Transactions,
    ROUND(COUNT(t.Transaction_ID) * 1.0 / COUNT(DISTINCT c.Customer_ID), 2) AS Avg_Tx_Per_User,
    ROUND(SUM(t.Amount_NGN), 2) AS Total_Spent_NGN
FROM FinTrust_Customer_Data c
LEFT JOIN FinTrust_Transaction_Data t ON c.Customer_ID = t.Customer_ID
GROUP BY 1
ORDER BY Avg_Tx_Per_User DESC;

-- 6. Geographic Distribution Across Major Banking Hubs
-- Business Interpretation: Lagos, Kano, and Port Harcourt concentrate over 62% of 
-- transaction volume, confirming primary regional hubs.
SELECT 
    c.City,
    COUNT(t.Transaction_ID) AS Tx_Volume,
    ROUND(SUM(t.Amount_NGN), 2) AS Total_Volume_NGN,
    ROUND(AVG(t.Amount_NGN), 2) AS Mean_Ticket_NGN
FROM FinTrust_Customer_Data c
JOIN FinTrust_Transaction_Data t ON c.Customer_ID = t.Customer_ID
GROUP BY c.City
ORDER BY Total_Volume_NGN DESC;

-- 7. Monthly Income Tier Analysis
-- Business Interpretation: The 500k-999k income cohort provides the largest aggregate 
-- monetary flow, while the 1m+ cohort generates higher per-transaction revenue.
SELECT 
    c.Monthly_Income_Band,
    COUNT(DISTINCT c.Customer_ID) AS User_Count,
    COUNT(t.Transaction_ID) AS Tx_Count,
    ROUND(SUM(t.Amount_NGN), 2) AS Total_Value_NGN,
    ROUND(AVG(t.Amount_NGN), 2) AS Avg_Value_NGN
FROM FinTrust_Customer_Data c
JOIN FinTrust_Transaction_Data t ON c.Customer_ID = t.Customer_ID
GROUP BY c.Monthly_Income_Band
ORDER BY Total_Value_NGN DESC;

-- 8. Top 10 High-Velocity Accounts
-- Business Interpretation: Identifies critical high-volume corporate and individual accounts 
-- requiring dedicated VIP relationship support and security monitoring.
SELECT 
    c.Customer_ID,
    c.Customer_Name,
    c.Customer_Segment,
    c.Account_Type,
    COUNT(t.Transaction_ID) AS Lifetime_Tx_Count,
    ROUND(SUM(t.Amount_NGN), 2) AS Total_Spent_NGN,
    SUM(CASE WHEN t.Risk_Review_Flag = 'Yes' THEN 1 ELSE 0 END) AS Total_Risk_Flags
FROM FinTrust_Customer_Data c
JOIN FinTrust_Transaction_Data t ON c.Customer_ID = t.Customer_ID
GROUP BY c.Customer_ID, c.Customer_Name, c.Customer_Segment, c.Account_Type
ORDER BY Total_Spent_NGN DESC
LIMIT 10;
