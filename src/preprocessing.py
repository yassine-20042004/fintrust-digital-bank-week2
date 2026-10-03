"""Enhanced Feature Engineering and Preprocessing Pipeline for FinTrust Week 3."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Income band numerical mapping helper
INCOME_BAND_MAP = {
    '<100k': 75000.0,
    '100k-249k': 175000.0,
    '250k-499k': 375000.0,
    '500k-999k': 750000.0,
    '1M+': 1500000.0,
    'Unknown': 200000.0
}


def engineer_features(df_tx: pd.DataFrame, df_cust: pd.DataFrame) -> pd.DataFrame:
    """Combines transactions with customer records and engineers 14 refined features for risk prediction."""
    merged = df_tx.merge(df_cust, on='Customer_ID', how='left')

    # Convert Transaction_DateTime if string
    if not pd.api.types.is_datetime64_any_dtype(merged['Transaction_DateTime']):
        merged['Transaction_DateTime'] = pd.to_datetime(merged['Transaction_DateTime'])

    # Baseline Features (1-8)
    merged['Tx_Hour'] = merged['Transaction_DateTime'].dt.hour
    merged['Tx_DayOfWeek'] = merged['Transaction_DateTime'].dt.dayofweek
    merged['Is_Weekend'] = merged['Tx_DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    merged['Log_Amount'] = np.log1p(merged['Amount_NGN'])
    merged['Is_International'] = (merged['International_Transaction'] == 'Yes').astype(int)
    merged['Cust_Lifetime_Tx'] = merged.groupby('Customer_ID')['Transaction_ID'].transform('count')

    cust_mean = merged.groupby('Customer_ID')['Amount_NGN'].transform('mean')
    merged['Amount_to_Mean_Ratio'] = merged['Amount_NGN'] / (cust_mean + 1e-5)

    q85 = merged['Amount_NGN'].quantile(0.85)
    merged['Is_High_Value'] = (merged['Amount_NGN'] > q85).astype(int)

    # Refined Features (9-14)
    # Feature 9: Late Night Transaction Binary Indicator (22:00 - 05:00)
    merged['Is_Night_Tx'] = merged['Tx_Hour'].apply(lambda h: 1 if (h >= 22 or h < 5) else 0)

    # Feature 10: Interaction between International Transaction and Weekend
    merged['International_Weekend_Interact'] = merged['Is_International'] * merged['Is_Weekend']

    # Feature 11: Transaction Amount relative to Estimated Monthly Income
    est_income = merged['Monthly_Income_Band'].map(INCOME_BAND_MAP).fillna(200000.0)
    merged['Amount_to_Income_Ratio'] = merged['Amount_NGN'] / (est_income + 1e-5)

    # Feature 12: Ratio of Customer Tenure to Customer Age (Loyalty Index)
    merged['Tenure_to_Age_Ratio'] = merged['Tenure_Months'] / (merged['Age'] * 12.0 + 1e-5)

    # Feature 13: Customer Failed Transaction Ratio
    merged['Is_Failed_Int'] = (merged['Transaction_Status'] == 'Failed').astype(int)
    merged['Cust_Failed_Tx_Ratio'] = merged.groupby('Customer_ID')['Is_Failed_Int'].transform('mean')
    merged.drop(columns=['Is_Failed_Int'], inplace=True)

    # Feature 14: Channel Historical Risk Score
    channel_risk = merged.groupby('Channel')['Is_International'].transform('mean')
    merged['Channel_Risk_Rate'] = channel_risk

    return merged


def get_preprocessor():
    """Returns fitted ColumnTransformer for numerical and categorical features."""
    numeric_features = [
        'Amount_NGN', 'Log_Amount', 'Age', 'Tenure_Months',
        'Digital_Engagement_Score', 'Tx_Hour', 'Tx_DayOfWeek',
        'Is_Weekend', 'Is_International', 'Cust_Lifetime_Tx',
        'Amount_to_Mean_Ratio', 'Is_High_Value', 'Is_Night_Tx',
        'International_Weekend_Interact', 'Amount_to_Income_Ratio',
        'Tenure_to_Age_Ratio', 'Cust_Failed_Tx_Ratio', 'Channel_Risk_Rate'
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
