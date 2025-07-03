
"""
Data Cleaning Module for CLV Prediction Project
This module handles data preprocessing and cleaning for the Online Retail Dataset
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """
    Load the online retail dataset

    Parameters:
    file_path (str): Path to the CSV file

    Returns:
    pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    """
    Clean the dataset by removing invalid records and creating new features

    Parameters:
    df (pd.DataFrame): Raw dataset

    Returns:
    pd.DataFrame: Cleaned dataset
    """
    initial_shape = df.shape
    print(f"Initial dataset shape: {initial_shape}")

    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Remove rows with missing CustomerID
    df = df.dropna(subset=['Customer ID'])
    print(f"After removing missing Customer ID: {df.shape} (removed {initial_shape[0] - df.shape[0]} rows)")

    # Filter out canceled orders (InvoiceNo starting with 'C')
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    print(f"After removing canceled orders: {df.shape}")

    # Remove negative or zero quantities and prices
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
    print(f"After removing negative/zero quantities and prices: {df.shape}")

    # Create TotalPrice column
    df['TotalPrice'] = df['Quantity'] * df['Price']

    # Remove outliers (optional - can be adjusted)
    Q1 = df['TotalPrice'].quantile(0.25)
    Q3 = df['TotalPrice'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_removed = df.shape[0]
    df = df[(df['TotalPrice'] >= lower_bound) & (df['TotalPrice'] <= upper_bound)]
    outliers_removed = outliers_removed - df.shape[0]
    print(f"After removing outliers: {df.shape} (removed {outliers_removed} outliers)")

    print(f"Final cleaned dataset shape: {df.shape}")
    print(f"Total data reduction: {((initial_shape[0] - df.shape[0]) / initial_shape[0] * 100):.2f}%")

    return df

def get_data_summary(df):
    """
    Generate a summary of the cleaned dataset

    Parameters:
    df (pd.DataFrame): Cleaned dataset

    Returns:
    dict: Summary statistics
    """
    summary = {
        'total_transactions': len(df),
        'unique_customers': df['Customer ID'].nunique(),
        'unique_products': df['StockCode'].nunique(),
        'date_range': f"{df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}",
        'total_revenue': df['TotalPrice'].sum(),
        'avg_order_value': df['TotalPrice'].mean(),
        'countries': df['Country'].nunique()
    }

    print("=== Dataset Summary ===")
    for key, value in summary.items():
        if key == 'total_revenue':
            print(f"{key.replace('_', ' ').title()}: ${value:,.2f}")
        elif key == 'avg_order_value':
            print(f"{key.replace('_', ' ').title()}: ${value:.2f}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")

    return summary

if __name__ == "__main__":
    # Load and clean data
    df = load_data('../data/online_retail_dataset.csv')
    if df is not None:
        cleaned_df = clean_data(df)
        summary = get_data_summary(cleaned_df)

        # Save cleaned data
        cleaned_df.to_csv('../data/cleaned_retail_data.csv', index=False)
        print("\nCleaned data saved to ../data/cleaned_retail_data.csv")
