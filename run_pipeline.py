#!/usr/bin/env python3
"""
Main Pipeline Script for CLV Prediction Project
This script runs the complete CLV prediction pipeline
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add src directory to path
sys.path.append('src')

def main():
    """
    Main function to run the CLV prediction pipeline
    """
    print("="*60)
    print("🚀 CUSTOMER LIFETIME VALUE PREDICTION PIPELINE")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        # Step 1: Data Cleaning
        print("📊 Step 1: Data Cleaning")
        print("-" * 30)
        from src.data_cleaning import load_data, clean_data, get_data_summary

        # Load data
        df = load_data('data/online_retail_dataset.csv')
        if df is None:
            print("❌ Error: Could not load dataset")
            return False

        # Clean data
        cleaned_df = clean_data(df)
        summary = get_data_summary(cleaned_df)

        # Save cleaned data
        cleaned_df.to_csv('data/cleaned_retail_data.csv', index=False)
        print("✅ Data cleaning completed\n")

        # Step 2: Feature Engineering
        print("🔧 Step 2: Feature Engineering (RFM Analysis)")
        print("-" * 30)
        from feature_engineering import (
            calculate_rfm_metrics, 
            create_rfm_scores, 
            prepare_features_for_modeling,
            create_target_variable
        )

        # Calculate RFM metrics
        rfm_metrics = calculate_rfm_metrics(cleaned_df)
        rfm_with_scores = create_rfm_scores(rfm_metrics)
        modeling_features = prepare_features_for_modeling(rfm_with_scores)
        final_data = create_target_variable(cleaned_df, rfm_with_scores)

        # Save feature engineering results
        rfm_with_scores.to_csv('data/rfm_analysis.csv', index=False)
        modeling_features.to_csv('data/modeling_features.csv', index=False)
        final_data.to_csv('data/clv_dataset.csv', index=False)
        print("✅ Feature engineering completed\n")

        # Step 3: Model Training
        print("🤖 Step 3: Model Training and Evaluation")
        print("-" * 30)
        from modeling import CLVModelTrainer

        # Initialize trainer
        trainer = CLVModelTrainer()

        # Prepare data
        X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = trainer.prepare_data(final_data)

        # Train models
        trainer.train_models(X_train, y_train)

        # Evaluate models
        results = trainer.evaluate_models(X_test, y_test)

        # Get feature importance
        feature_importance = trainer.get_feature_importance('Random Forest')

        # Save models and results
        trainer.save_models()
        # results.to_csv('outputs/model_evaluation_results.csv', index=False)
        # if feature_importance is not None:
        #     feature_importance.to_csv('outputs/feature_importance.csv', index=False)
        
        # Create the results directory if it doesn't exist
        results_path = os.path.join('outputs', 'results')
        os.makedirs(results_path, exist_ok=True)

        # Save evaluation results
        results_file = os.path.join(results_path, 'model_evaluation_results.csv')
        results.to_csv(results_file, index=False)
        print(f"✅ Saved model evaluation results to {results_file}")

        # Save feature importance if available
        if feature_importance is not None:
            importance_file = os.path.join(results_path, 'feature_importance.csv')
            feature_importance.to_csv(importance_file, index=False)
            print(f"✅ Saved feature importance to {importance_file}")

        print("✅ Model training completed\n")

        # Step 4: Visualization
        print("📈 Step 4: Creating Visualizations")
        print("-" * 30)
        from visualization import CLVVisualizer, create_summary_dashboard

        # Initialize visualizer
        visualizer = CLVVisualizer()

        # Create plots
        visualizer.plot_rfm_distribution(rfm_with_scores, save_plot=True)
        visualizer.plot_customer_segments(rfm_with_scores, save_plot=True)
        visualizer.plot_top_customers(rfm_with_scores, save_plot=True)
        visualizer.plot_model_comparison(results, save_plot=True)

        if feature_importance is not None:
            visualizer.plot_feature_importance(feature_importance, save_plot=True)

        # 🔧 Merge CLV before summary dashboard
        rfm_with_scores = rfm_with_scores.merge(
            final_data[['Customer_ID', 'CLV_Target']],
            on='Customer_ID',
            how='left'
        )
        # Create comprehensive dashboard
        create_summary_dashboard(rfm_with_scores, results, save_plot=True)

        print("✅ Visualizations completed\n")

        # Step 5: Summary
        print("📋 Step 5: Pipeline Summary")
        print("-" * 30)

        # Display key results
        best_model = results.loc[results['R²'].idxmax()]
        print(f"🏆 Best Model: {best_model['Model']}")
        print(f"📊 R² Score: {best_model['R²']:.4f}")
        print(f"📏 RMSE: {best_model['RMSE']:.4f}")
        print(f"📐 MAE: {best_model['MAE']:.4f}")
        print()

        print(f"👥 Total Customers: {len(rfm_with_scores):,}")
        print(f"💰 Total Revenue: ${rfm_with_scores['Monetary'].sum():,.2f}")
        print(f"🎯 Average CLV: ${final_data['CLV_Target'].mean():.2f}")
        print()

        segment_counts = rfm_with_scores['Customer_Segment'].value_counts()
        print("🏷️ Customer Segments:")
        for segment, count in segment_counts.items():
            percentage = (count / len(rfm_with_scores)) * 100
            print(f"   • {segment}: {count} customers ({percentage:.1f}%)")

        print()
        print("📁 Generated Files:")
        print("   • data/cleaned_retail_data.csv")
        print("   • data/rfm_analysis.csv")
        print("   • data/clv_dataset.csv")
        print("   • outputs/results/model_evaluation_results.csv")
        print("   • outputs/results/feature_importance.csv")
        print("   • outputs/plots/ (multiple visualization files)")
        print("   • outputs/models/ (trained models and scalers)")
        print()

        print("🎉 Pipeline completed successfully!")
        print("🔗 Run 'streamlit run streamlit_app.py' to launch the dashboard")

        return True

    except Exception as e:
        print(f"❌ Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        print()
        print("="*60)
        print(f"Pipeline finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
