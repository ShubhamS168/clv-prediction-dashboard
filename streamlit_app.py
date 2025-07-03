
"""
Streamlit Dashboard for CLV Prediction
This module creates an interactive web dashboard for CLV prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="CLV Prediction Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1E88E5;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
# .prediction-box {
#     background-color: #c8f7c5;
#     padding: 1.5rem;
#     border-radius: 0.5rem;
#     border-left: 5px solid #4CAF50;
# }
.prediction-box {
    background-color: #c8f7c5; /* darker green */
    color: #000000; /* set text color to black */
    padding: 1.5rem;
    border-radius: 0.5rem;
    border-left: 5px solid #4CAF50;
}
</style>
""", unsafe_allow_html=True)

class CLVDashboard:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_names = None
        self.load_models()

    def load_models(self):
        """Load trained models and preprocessing objects"""
        try:
            # Load models
            model_names = ['random_forest', 'linear_regression', 'xgboost', 'gradient_boosting']
            for name in model_names:
                try:
                    with open(f'outputs/models/clv_model_{name}.pkl', 'rb') as f:
                        self.models[name] = pickle.load(f)
                except FileNotFoundError:
                    continue

            # Load scaler
            with open('outputs/models/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)

            # Load feature names
            with open('outputs/models/feature_names.pkl', 'rb') as f:
                self.feature_names = pickle.load(f)

        except Exception as e:
            st.error(f"Error loading models: {e}")
            st.info("Please run the modeling script first to train the models.")

    def predict_clv(self, features, model_name):
        """Predict CLV for given features"""
        if model_name not in self.models:
            return None

        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # Make prediction
        prediction = self.models[model_name].predict(features_scaled)[0]

        return max(0, prediction)  # Ensure non-negative CLV

    def create_radar_chart(self, rfm_values):
        """Create radar chart for RFM values"""
        categories = ['Recency Score', 'Frequency Score', 'Monetary Score']

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=rfm_values,
            theta=categories,
            fill='toself',
            name='Customer Profile',
            line_color='blue'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 5]
                )),
            showlegend=True,
            title="Customer RFM Profile"
        )

        return fig

    def run_dashboard(self):
        """Main dashboard function"""
        # Header
        st.markdown('<h1 class="main-header">💰 Customer Lifetime Value Prediction Dashboard</h1>', 
                   unsafe_allow_html=True)

        # Sidebar
        st.sidebar.title("Navigation")
        # page = st.sidebar.selectbox("Choose a page", 
        #                            ["CLV Prediction", "Data Analysis", "Model Performance"])
        page = st.sidebar.radio("", 
                        ["CLV Prediction", "Data Analysis", "Model Performance"])

        if page == "CLV Prediction":
            self.clv_prediction_page()
        elif page == "Data Analysis":
            self.data_analysis_page()
        elif page == "Model Performance":
            self.model_performance_page()

    def clv_prediction_page(self):
        """CLV Prediction page"""
        st.header("🎯 Customer Lifetime Value Prediction")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Customer Input")

            # RFM inputs
            st.markdown("**RFM Metrics**")
            recency = st.slider("Recency (days since last purchase)", 0, 365, 30)
            frequency = st.slider("Frequency (number of orders)", 1, 50, 5)
            monetary = st.slider("Monetary (total spent $)", 0, 5000, 500)

            # RFM Scores
            st.markdown("**RFM Scores (1-5)**")
            r_score = st.slider("Recency Score", 1, 5, 3)
            f_score = st.slider("Frequency Score", 1, 5, 3)
            m_score = st.slider("Monetary Score", 1, 5, 3)

            # Additional features
            st.markdown("**Additional Features**")
            total_quantity = st.number_input("Total Quantity Purchased", 0, 1000, 50)
            avg_quantity = st.number_input("Average Quantity per Order", 0.0, 50.0, 3.0)
            avg_order_value = st.number_input("Average Order Value ($)", 0.0, 1000.0, 100.0)
            unique_products = st.number_input("Unique Products Purchased", 1, 100, 10)
            tenure_days = st.number_input("Customer Tenure (days)", 0, 1000, 90)

            # Model selection
            available_models = list(self.models.keys())
            if available_models:
                selected_model = st.selectbox("Select Model", available_models)
            else:
                st.error("No models available. Please train models first.")
                return

        with col2:
            st.subheader("Prediction Results")

            if st.button("Predict CLV", type="primary"):
                if self.models and self.scaler and self.feature_names:
                    # Prepare features
                    # Step 1: Prepare input feature dictionary
                    input_dict = {
                        'Recency': recency,
                        'Frequency': frequency,
                        'Monetary': monetary,
                        'Recency Score': r_score,
                        'Frequency Score': f_score,
                        'Monetary Score': m_score,
                        'Total_Quantity': total_quantity,
                        'Avg_Quantity': avg_quantity,
                        'Avg_Order_Value': avg_order_value,
                        'Order_Value_Std': 0,
                        'Unique_Products': unique_products,
                        'Tenure_Days': tenure_days,
                        'Purchase_Frequency_Rate': frequency / (tenure_days + 1),
                        'Avg_Monetary_per_Order': monetary / frequency if frequency != 0 else 0,
                        'Quantity_per_Product': total_quantity / unique_products if unique_products != 0 else 0
                    }

                    # Step 2: Fill in any missing model features with default 0
                    for fname in self.feature_names:
                        if fname not in input_dict:
                            input_dict[fname] = 0

                    # Step 3: Ensure correct order and structure
                    input_df = pd.DataFrame([input_dict])[self.feature_names]
                    features = input_df.values.flatten()

                    # Make prediction
                    clv_prediction = self.predict_clv(features, selected_model)

                    if clv_prediction is not None:
                        # Display prediction
                        st.markdown(f"""
                        <div class="prediction-box">
                        <h3>Predicted CLV: ${clv_prediction:.2f}</h3>
                        <p>Model: {selected_model.replace('_', ' ').title()}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Customer segment
                        if clv_prediction > 1000:
                            segment = "High Value"
                            color = "green"
                        elif clv_prediction > 500:
                            segment = "Medium Value"
                            color = "orange"
                        else:
                            segment = "Low Value"
                            color = "red"

                        st.markdown(f"**Customer Segment:** :{color}[{segment}]")

                        # Radar chart
                        radar_fig = self.create_radar_chart([r_score, f_score, m_score])
                        st.plotly_chart(radar_fig, use_container_width=True)

                        # Recommendations
                        st.subheader("📈 Recommendations")
                        if segment == "High Value":
                            st.success("🌟 VIP Customer - Focus on retention and upselling")
                        elif segment == "Medium Value":
                            st.warning("📊 Potential Growth - Target with personalized offers")
                        else:
                            st.info("🎯 Re-engagement - Consider win-back campaigns")
                    else:
                        st.error("Error making prediction")
                else:
                    st.error("Models not loaded properly")

    def data_analysis_page(self):
        """Data Analysis page"""
        st.header("📊 Data Analysis")

        try:
            # Load data
            rfm_df = pd.read_csv('data/rfm_analysis.csv')
            clv_df = pd.read_csv('data/clv_dataset.csv')

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Customers", len(rfm_df))
            with col2:
                st.metric("Total Revenue", f"${rfm_df['Monetary'].sum():,.2f}")
            with col3:
                st.metric("Average CLV", f"${clv_df['CLV_Target'].mean():.2f}")
            with col4:
                st.metric("Average Orders", f"{rfm_df['Frequency'].mean():.1f}")

            # Charts
            col1, col2 = st.columns(2)

            with col1:
                # Customer segments pie chart
                fig_pie = px.pie(rfm_df, names='Customer_Segment', title='Customer Segments')
                st.plotly_chart(fig_pie, use_container_width=True)

                # RFM correlation heatmap
                rfm_corr = rfm_df[['Recency', 'Frequency', 'Monetary']].corr()
                fig_heatmap = px.imshow(rfm_corr, text_auto=True, title='RFM Correlation')
                st.plotly_chart(fig_heatmap, use_container_width=True)

            with col2:
                # CLV distribution
                fig_hist = px.histogram(clv_df, x='CLV_Target', title='CLV Distribution', nbins=20)
                st.plotly_chart(fig_hist, use_container_width=True)

                # Top customers
                top_customers = rfm_df.nlargest(10, 'Monetary')
                fig_bar = px.bar(top_customers, x='Customer_ID', y='Monetary', 
                               title='Top 10 Customers by Revenue')
                st.plotly_chart(fig_bar, use_container_width=True)

            # Detailed data table
            st.subheader("Customer Data")
            st.dataframe(rfm_df.head(20), use_container_width=True)

        except FileNotFoundError:
            st.error("Data files not found. Please run the data preprocessing scripts first.")

    def model_performance_page(self):
        """Model Performance page"""
        st.header("🤖 Model Performance")

        try:
            # Load model results
            results_df = pd.read_csv('outputs/model_evaluation_results.csv')

            # Performance metrics
            st.subheader("Model Comparison")

            col1, col2, col3 = st.columns(3)

            with col1:
                fig_rmse = px.bar(results_df, x='Model', y='RMSE', title='RMSE Comparison')
                st.plotly_chart(fig_rmse, use_container_width=True)

            with col2:
                fig_mae = px.bar(results_df, x='Model', y='MAE', title='MAE Comparison')
                st.plotly_chart(fig_mae, use_container_width=True)

            with col3:
                fig_r2 = px.bar(results_df, x='Model', y='R²', title='R² Score Comparison')
                st.plotly_chart(fig_r2, use_container_width=True)

            # Best model
            best_model = results_df.loc[results_df['R²'].idxmax()]
            st.success(f"🏆 Best Model: {best_model['Model']} (R² = {best_model['R²']:.3f})")

            # Detailed results table
            st.subheader("Detailed Results")
            st.dataframe(results_df, use_container_width=True)

            # Feature importance
            try:
                feature_importance = pd.read_csv('outputs/feature_importance.csv')
                st.subheader("Feature Importance")

                fig_importance = px.bar(feature_importance.head(10), 
                                      x='Importance', y='Feature', 
                                      orientation='h', title='Top 10 Feature Importance')
                st.plotly_chart(fig_importance, use_container_width=True)

            except FileNotFoundError:
                st.warning("Feature importance data not found.")

        except FileNotFoundError:
            st.error("Model results not found. Please run the modeling script first.")

# Main execution
if __name__ == "__main__":
    dashboard = CLVDashboard()
    dashboard.run_dashboard()
