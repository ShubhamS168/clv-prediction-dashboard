
"""
Feature Engineering Module for CLV Prediction Project
This module handles RFM analysis and feature creation for CLV modeling
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def calculate_rfm_metrics(df, analysis_date=None):
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics for each customer

    Parameters:
    df (pd.DataFrame): Cleaned transaction data
    analysis_date (datetime): Reference date for recency calculation

    Returns:
    pd.DataFrame: RFM metrics for each customer
    """
    if analysis_date is None:
        analysis_date = df['InvoiceDate'].max() + timedelta(days=1)

    print(f"Calculating RFM metrics with analysis date: {analysis_date}")

    # Group by customer and calculate metrics
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (analysis_date - x.max()).days,  # Recency
        'Invoice': 'nunique',  # Frequency (number of unique orders)
        'TotalPrice': 'sum'  # Monetary (total spent)
    }).reset_index()

    # Rename columns
    rfm.columns = ['Customer_ID', 'Recency', 'Frequency', 'Monetary']

    # Calculate additional features
    customer_stats = df.groupby('Customer ID').agg({
        'Quantity': ['sum', 'mean'],
        'TotalPrice': ['mean', 'std'],
        'StockCode': 'nunique',
        'InvoiceDate': ['min', 'max']
    }).reset_index()

    # Flatten column names
    customer_stats.columns = [
        'Customer_ID', 'Total_Quantity', 'Avg_Quantity_Per_Order',
        'Avg_Order_Value', 'Order_Value_Std', 'Unique_Products',
        'First_Purchase', 'Last_Purchase'
    ]

    # Calculate customer tenure (days between first and last purchase)
    customer_stats['Tenure_Days'] = (customer_stats['Last_Purchase'] - customer_stats['First_Purchase']).dt.days
    customer_stats['Tenure_Days'] = customer_stats['Tenure_Days'].fillna(0)  # For single-purchase customers

    # Merge RFM with additional features
    rfm_features = pd.merge(rfm, customer_stats, on='Customer_ID', how='left')

    # Fill NaN standard deviation with 0 (customers with only one order)
    rfm_features['Order_Value_Std'] = rfm_features['Order_Value_Std'].fillna(0)

    # Calculate purchase frequency rate (orders per day)
    rfm_features['Purchase_Frequency_Rate'] = np.where(
        rfm_features['Tenure_Days'] > 0,
        rfm_features['Frequency'] / rfm_features['Tenure_Days'],
        rfm_features['Frequency']  # For single-day customers
    )

    print(f"RFM metrics calculated for {len(rfm_features)} customers")

    return rfm_features

def create_rfm_scores(rfm_df):
    """
    Create RFM scores and segments based on quantiles

    Parameters:
    rfm_df (pd.DataFrame): RFM metrics dataframe

    Returns:
    pd.DataFrame: RFM dataframe with scores and segments
    """
    rfm_scores = rfm_df.copy()

    # Create RFM scores (1-5 scale)
    # For Recency: lower is better (recent customers)
    rfm_scores['R_Score'] = pd.qcut(rfm_scores['Recency'].rank(method='first'), 5, labels=[5,4,3,2,1])

    # For Frequency and Monetary: higher is better
    rfm_scores['F_Score'] = pd.qcut(rfm_scores['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    rfm_scores['M_Score'] = pd.qcut(rfm_scores['Monetary'].rank(method='first'), 5, labels=[1,2,3,4,5])

    # Convert to numeric
    rfm_scores['R_Score'] = rfm_scores['R_Score'].astype(int)
    rfm_scores['F_Score'] = rfm_scores['F_Score'].astype(int)
    rfm_scores['M_Score'] = rfm_scores['M_Score'].astype(int)

    # Create RFM segment
    rfm_scores['RFM_Score'] = rfm_scores['R_Score'].astype(str) + rfm_scores['F_Score'].astype(str) + rfm_scores['M_Score'].astype(str)

    # Create customer segments based on RFM scores
    def segment_customers(row):
        if row['RFM_Score'] in ['555', '554', '544', '545', '454', '455', '445']:
            return 'Champions'
        elif row['RFM_Score'] in ['543', '444', '435', '355', '354', '345', '344', '335']:
            return 'Loyal Customers'
        elif row['RFM_Score'] in ['512', '511', '422', '421', '412', '411', '311']:
            return 'Potential Loyalists'
        elif row['RFM_Score'] in ['512', '511', '331', '321', '312', '231']:
            return 'New Customers'
        elif row['RFM_Score'] in ['155', '154', '144', '214', '215', '115', '114']:
            return 'At Risk'
        elif row['RFM_Score'] in ['155', '154', '144', '214', '215', '115']:
            return 'Cannot Lose Them'
        elif row['RFM_Score'] in ['111', '112', '121', '131', '141', '151']:
            return 'Lost'
        else:
            return 'Others'

    rfm_scores['Customer_Segment'] = rfm_scores.apply(segment_customers, axis=1)

    return rfm_scores

def prepare_features_for_modeling(rfm_df):
    """
    Prepare features for machine learning modeling

    Parameters:
    rfm_df (pd.DataFrame): RFM dataframe with scores

    Returns:
    pd.DataFrame: Features ready for modeling
    """
    # Select features for modeling
    feature_columns = [
        'Recency', 'Frequency', 'Monetary', 'R_Score', 'F_Score', 'M_Score',
        'Total_Quantity', 'Avg_Quantity_Per_Order', 'Avg_Order_Value',
        'Order_Value_Std', 'Unique_Products', 'Tenure_Days', 'Purchase_Frequency_Rate'
    ]

    modeling_df = rfm_df[['Customer_ID'] + feature_columns].copy()

    # Handle any remaining missing values
    modeling_df = modeling_df.fillna(0)

    # Create additional derived features
    modeling_df['Monetary_per_Frequency'] = modeling_df['Monetary'] / modeling_df['Frequency']
    modeling_df['Quantity_per_Product'] = modeling_df['Total_Quantity'] / modeling_df['Unique_Products']

    return modeling_df

def create_target_variable(df, rfm_df, prediction_period_days=90):
    """
    Create target variable for CLV prediction
    Note: This is a simplified version since we have limited historical data

    Parameters:
    df (pd.DataFrame): Original transaction data
    rfm_df (pd.DataFrame): RFM metrics
    prediction_period_days (int): Period for future CLV prediction

    Returns:
    pd.DataFrame: RFM dataframe with target variable
    """
    # For this example, we'll create a proxy target based on existing data
    # In real scenarios, you would use future purchase data

    # Calculate average order value and frequency for each customer
    avg_metrics = rfm_df.copy()

    # Create a simple CLV proxy: Monetary * (Frequency/Recency) * scaling factor
    # This is a simplified approach for demonstration
    avg_metrics['CLV_Target'] = (
        avg_metrics['Monetary'] * 
        (avg_metrics['Frequency'] / (avg_metrics['Recency'] + 1)) * 
        (prediction_period_days / 365)
    )

    # Ensure non-negative values
    avg_metrics['CLV_Target'] = np.maximum(avg_metrics['CLV_Target'], 0)

    print(f"Target variable created. Mean CLV: ${avg_metrics['CLV_Target'].mean():.2f}")
    print(f"CLV range: ${avg_metrics['CLV_Target'].min():.2f} - ${avg_metrics['CLV_Target'].max():.2f}")

    return avg_metrics

if __name__ == "__main__":
    # Load cleaned data
    df = pd.read_csv('../data/cleaned_retail_data.csv')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Calculate RFM metrics
    rfm_metrics = calculate_rfm_metrics(df)

    # Create RFM scores and segments
    rfm_with_scores = create_rfm_scores(rfm_metrics)

    # Prepare features for modeling
    modeling_features = prepare_features_for_modeling(rfm_with_scores)

    # Create target variable
    final_data = create_target_variable(df, rfm_with_scores)

    # Save results
    rfm_with_scores.to_csv('../data/rfm_analysis.csv', index=False)
    modeling_features.to_csv('../data/modeling_features.csv', index=False)
    final_data.to_csv('../data/clv_dataset.csv', index=False)

    print("\nFeature engineering completed!")
    print("Files saved:")
    print("- ../data/rfm_analysis.csv")
    print("- ../data/modeling_features.csv") 
    print("- ../data/clv_dataset.csv")
