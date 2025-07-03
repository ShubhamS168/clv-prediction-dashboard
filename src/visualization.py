
"""
Visualization Module for CLV Prediction Project
This module handles data visualization and plotting for CLV analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

class CLVVisualizer:
    """
    Class to handle CLV data visualization
    """

    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        self.save_path = '../outputs/plots/'

    def plot_rfm_distribution(self, rfm_df, save_plot=True):
        """
        Plot RFM metrics distribution

        Parameters:
        rfm_df (pd.DataFrame): RFM dataframe
        save_plot (bool): Whether to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RFM Metrics Distribution', fontsize=16, fontweight='bold')

        # Recency distribution
        axes[0, 0].hist(rfm_df['Recency'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Recency Distribution')
        axes[0, 0].set_xlabel('Days Since Last Purchase')
        axes[0, 0].set_ylabel('Number of Customers')

        # Frequency distribution
        axes[0, 1].hist(rfm_df['Frequency'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('Frequency Distribution')
        axes[0, 1].set_xlabel('Number of Orders')
        axes[0, 1].set_ylabel('Number of Customers')

        # Monetary distribution
        axes[1, 0].hist(rfm_df['Monetary'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 0].set_title('Monetary Distribution')
        axes[1, 0].set_xlabel('Total Spent ($)')
        axes[1, 0].set_ylabel('Number of Customers')

        # RFM correlation heatmap
        rfm_corr = rfm_df[['Recency', 'Frequency', 'Monetary']].corr()
        sns.heatmap(rfm_corr, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
        axes[1, 1].set_title('RFM Correlation Heatmap')

        plt.tight_layout()

        if save_plot:
            plt.savefig(f'{self.save_path}rfm_distribution.png', dpi=300, bbox_inches='tight')
            print(f"RFM distribution plot saved to {self.save_path}rfm_distribution.png")

        plt.show()

    def plot_customer_segments(self, rfm_df, save_plot=True):
        """
        Plot customer segmentation

        Parameters:
        rfm_df (pd.DataFrame): RFM dataframe with segments
        save_plot (bool): Whether to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Customer Segmentation Analysis', fontsize=16, fontweight='bold')

        # Segment distribution
        segment_counts = rfm_df['Customer_Segment'].value_counts()
        axes[0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0].set_title('Customer Segment Distribution')

        # Segment value contribution
        segment_value = rfm_df.groupby('Customer_Segment')['Monetary'].sum().sort_values(ascending=False)
        axes[1].bar(segment_value.index, segment_value.values, color='lightblue', edgecolor='black')
        axes[1].set_title('Revenue Contribution by Segment')
        axes[1].set_xlabel('Customer Segment')
        axes[1].set_ylabel('Total Revenue ($)')
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_plot:
            plt.savefig(f'{self.save_path}customer_segments.png', dpi=300, bbox_inches='tight')
            print(f"Customer segments plot saved to {self.save_path}customer_segments.png")

        plt.show()

    def plot_top_customers(self, rfm_df, top_n=10, save_plot=True):
        """
        Plot top customers by monetary value

        Parameters:
        rfm_df (pd.DataFrame): RFM dataframe
        top_n (int): Number of top customers to show
        save_plot (bool): Whether to save the plot
        """
        top_customers = rfm_df.nlargest(top_n, 'Monetary')

        plt.figure(figsize=self.figsize)
        bars = plt.bar(range(len(top_customers)), top_customers['Monetary'], 
                      color='gold', edgecolor='black', alpha=0.8)

        plt.title(f'Top {top_n} Customers by Total Spend', fontsize=14, fontweight='bold')
        plt.xlabel('Customer Rank')
        plt.ylabel('Total Spent ($)')
        plt.xticks(range(len(top_customers)), [f'C{int(id)}' for id in top_customers['Customer_ID']], rotation=45)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'${height:.0f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        if save_plot:
            plt.savefig(f'{self.save_path}top_customers.png', dpi=300, bbox_inches='tight')
            print(f"Top customers plot saved to {self.save_path}top_customers.png")

        plt.show()

    def plot_model_comparison(self, results_df, save_plot=True):
        """
        Plot model comparison results

        Parameters:
        results_df (pd.DataFrame): Model evaluation results
        save_plot (bool): Whether to save the plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

        # RMSE comparison
        axes[0].bar(results_df['Model'], results_df['RMSE'], color='lightcoral', edgecolor='black')
        axes[0].set_title('Root Mean Square Error (RMSE)')
        axes[0].set_ylabel('RMSE')
        axes[0].tick_params(axis='x', rotation=45)

        # MAE comparison
        axes[1].bar(results_df['Model'], results_df['MAE'], color='lightblue', edgecolor='black')
        axes[1].set_title('Mean Absolute Error (MAE)')
        axes[1].set_ylabel('MAE')
        axes[1].tick_params(axis='x', rotation=45)

        # R² comparison
        axes[2].bar(results_df['Model'], results_df['R²'], color='lightgreen', edgecolor='black')
        axes[2].set_title('R² Score')
        axes[2].set_ylabel('R² Score')
        axes[2].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_plot:
            plt.savefig(f'{self.save_path}model_comparison.png', dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to {self.save_path}model_comparison.png")

        plt.show()

    def plot_feature_importance(self, importance_df, top_n=10, save_plot=True):
        """
        Plot feature importance

        Parameters:
        importance_df (pd.DataFrame): Feature importance dataframe
        top_n (int): Number of top features to show
        save_plot (bool): Whether to save the plot
        """
        top_features = importance_df.head(top_n)

        plt.figure(figsize=self.figsize)
        bars = plt.barh(range(len(top_features)), top_features['Importance'], 
                       color='steelblue', edgecolor='black', alpha=0.8)

        plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.yticks(range(len(top_features)), top_features['Feature'])

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + width*0.005, bar.get_y() + bar.get_height()/2.,
                    f'{width:.3f}', ha='left', va='center', fontweight='bold')

        plt.gca().invert_yaxis()  # Highest importance at top
        plt.tight_layout()

        if save_plot:
            plt.savefig(f'{self.save_path}feature_importance.png', dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {self.save_path}feature_importance.png")

        plt.show()

    def plot_actual_vs_predicted(self, y_true, y_pred, model_name, save_plot=True):
        """
        Plot actual vs predicted values

        Parameters:
        y_true: Actual values
        y_pred: Predicted values
        model_name (str): Name of the model
        save_plot (bool): Whether to save the plot
        """
        plt.figure(figsize=self.figsize)

        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.6, color='steelblue', edgecolors='black', linewidth=0.5)

        # Perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        plt.xlabel('Actual CLV')
        plt.ylabel('Predicted CLV')
        plt.title(f'Actual vs Predicted CLV - {model_name}', fontsize=14, fontweight='bold')
        plt.legend()

        # Add R² score
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontweight='bold')

        plt.tight_layout()

        if save_plot:
            filename = f'{self.save_path}actual_vs_predicted_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Actual vs predicted plot saved to {filename}")

        plt.show()

    def plot_clv_distribution(self, clv_values, save_plot=True):
        """
        Plot CLV distribution

        Parameters:
        clv_values: Array of CLV values
        save_plot (bool): Whether to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Customer Lifetime Value Distribution', fontsize=16, fontweight='bold')

        # Histogram
        axes[0].hist(clv_values, bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[0].set_title('CLV Histogram')
        axes[0].set_xlabel('CLV ($)')
        axes[0].set_ylabel('Number of Customers')

        # Box plot
        axes[1].boxplot(clv_values, vert=True, patch_artist=True, 
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
        axes[1].set_title('CLV Box Plot')
        axes[1].set_ylabel('CLV ($)')

        plt.tight_layout()

        if save_plot:
            plt.savefig(f'{self.save_path}clv_distribution.png', dpi=300, bbox_inches='tight')
            print(f"CLV distribution plot saved to {self.save_path}clv_distribution.png")

        plt.show()

def create_summary_dashboard(rfm_df, results_df, save_plot=True):
    """
    Create a comprehensive dashboard with key metrics

    Parameters:
    rfm_df (pd.DataFrame): RFM dataframe
    results_df (pd.DataFrame): Model results
    save_plot (bool): Whether to save the plot
    """
    fig = plt.figure(figsize=(20, 15))

    # Create grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Customer segment pie chart
    ax1 = fig.add_subplot(gs[0, 0])
    segment_counts = rfm_df['Customer_Segment'].value_counts()
    ax1.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Customer Segments', fontweight='bold')

    # 2. RFM metrics summary
    ax2 = fig.add_subplot(gs[0, 1])
    rfm_stats = rfm_df[['Recency', 'Frequency', 'Monetary']].describe().loc[['mean', 'std']]
    sns.heatmap(rfm_stats, annot=True, fmt='.2f', cmap='viridis', ax=ax2)
    ax2.set_title('RFM Summary Statistics', fontweight='bold')

    # 3. Model performance
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(results_df['Model'], results_df['R²'], color='lightgreen', edgecolor='black')
    ax3.set_title('Model R² Scores', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)

    # 4. Top customers
    ax4 = fig.add_subplot(gs[1, :])
    top_customers = rfm_df.nlargest(10, 'Monetary')
    ax4.bar(range(len(top_customers)), top_customers['Monetary'], color='gold', edgecolor='black')
    ax4.set_title('Top 10 Customers by Revenue', fontweight='bold')
    ax4.set_xlabel('Customer Rank')
    ax4.set_ylabel('Total Spent ($)')
    ax4.set_xticks(range(len(top_customers)))
    ax4.set_xticklabels([f'C{int(id)}' for id in top_customers['Customer_ID']])

    # 5. CLV distribution
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.hist(rfm_df['CLV_Target'], bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax5.set_title('CLV Distribution', fontweight='bold')
    ax5.set_xlabel('CLV ($)')
    ax5.set_ylabel('Customers')

    # 6. Frequency vs Monetary scatter
    ax6 = fig.add_subplot(gs[2, 1])
    scatter = ax6.scatter(rfm_df['Frequency'], rfm_df['Monetary'], 
                         c=rfm_df['Recency'], cmap='viridis', alpha=0.6)
    ax6.set_title('Frequency vs Monetary (colored by Recency)', fontweight='bold')
    ax6.set_xlabel('Frequency')
    ax6.set_ylabel('Monetary')
    plt.colorbar(scatter, ax=ax6, label='Recency')

    # 7. Summary metrics
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    # Calculate summary metrics
    total_customers = len(rfm_df)
    total_revenue = rfm_df['Monetary'].sum()
    avg_clv = rfm_df['CLV_Target'].mean()
    best_model = results_df.loc[results_df['R²'].idxmax(), 'Model']
    best_r2 = results_df['R²'].max()

    summary_text = f"""
    SUMMARY METRICS

    Total Customers: {total_customers:,}
    Total Revenue: ${total_revenue:,.2f}
    Average CLV: ${avg_clv:.2f}

    Best Model: {best_model}
    Best R² Score: {best_r2:.3f}

    Top Segment: {segment_counts.index[0]}
    ({segment_counts.iloc[0]} customers)
    """

    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.suptitle('CLV Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)

    if save_plot:
        plt.savefig('../outputs/plots/clv_dashboard.png', dpi=300, bbox_inches='tight')
        print("Dashboard saved to ../outputs/plots/clv_dashboard.png")

    plt.show()

if __name__ == "__main__":
    # This script can be run independently to generate all visualizations
    print("Visualization module loaded successfully!")
    print("Use CLVVisualizer class to create visualizations.")
