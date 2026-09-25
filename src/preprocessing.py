"""Feature Engineering and Preprocessing Pipeline."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def engineer_features(df_tx: pd.DataFrame, df_cust: pd.DataFrame) -> pd.DataFrame:
    """Combines transactions with customer records and engineers 8 predictive features."""
    merged = df_tx.merge(df_cust, on='Customer_ID', how='left')

    # Feature 1: Transaction Hour
    merged['Tx_Hour'] = merged['Transaction_DateTime'].dt.hour
    # Feature 2: Day of Week
    merged['Tx_DayOfWeek'] = merged['Transaction_DateTime'].dt.dayofweek
    # Feature 3: Weekend Indicator
    merged['Is_Weekend'] = merged['Tx_DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    # Feature 4: Log Transformed Amount
    merged['Log_Amount'] = np.log1p(merged['Amount_NGN'])
    # Feature 5: Binary International Indicator
    merged['Is_International'] = (merged['International_Transaction'] == 'Yes').astype(int)
    # Feature 6: Customer Aggregate Transaction Count
    merged['Cust_Lifetime_Tx'] = merged.groupby('Customer_ID')['Transaction_ID'].transform('count')
    # Feature 7: Ratio of Transaction Amount to Customer Mean
    cust_mean = merged.groupby('Customer_ID')['Amount_NGN'].transform('mean')
    merged['Amount_to_Mean_Ratio'] = merged['Amount_NGN'] / (cust_mean + 1e-5)
    # Feature 8: High Value Outlier Indicator (>85th percentile)
    q85 = merged['Amount_NGN'].quantile(0.85)
    merged['Is_High_Value'] = (merged['Amount_NGN'] > q85).astype(int)

    return merged


def get_preprocessor():
    """Returns fitted ColumnTransformer for numerical and categorical features."""
    numeric_features = [
        'Amount_NGN', 'Log_Amount', 'Age', 'Tenure_Months',
        'Digital_Engagement_Score', 'Tx_Hour', 'Tx_DayOfWeek',
        'Is_Weekend', 'Is_International', 'Cust_Lifetime_Tx',
        'Amount_to_Mean_Ratio', 'Is_High_Value'
    ]
    categorical_features = [
        'Transaction_Type', 'Channel', 'Device_Type', 'Location',
        'Transaction_Status', 'Gender', 'City', 'Customer_Segment',
        'Account_Type', 'Monthly_Income_Band', 'Preferred_Channel'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    return preprocessor, numeric_features, categorical_features
