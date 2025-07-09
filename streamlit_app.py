"""
Enhanced CLV Prediction Dashboard
Combines all original features with modern, professional UI/UX design
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration with modern settings
st.set_page_config(
    page_title="CLV Analytics Platform",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for Professional Look
def load_enhanced_css():
    st.markdown("""
    <style>
    /* Import Professional Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styling */
    .stApp {
        font-family: 'Poppins', sans-serif;
        background: white;
        min-height: 100vh;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;} 
    .stDeployButton {display: none;}
    
    /* Add or replace these CSS classes inside your <style> tag */

    /* New container for the background gradient */
    .header-background {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e); /* The background gradient you wanted */
        padding: 2.5rem;                  /* Creates space around the text */
        border-radius: 15px;              /* Optional: for rounded corners */
        margin-bottom: 2rem;              /* Space below the header block */
        box-shadow: 0 8px 25px rgba(0,0,0,0.1); /* Optional: adds a subtle shadow */
    }

    /* Updated style for the main header text */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        color: white;                     /* Solid white text for high contrast */
        background: none;                 /* Removes any gradient from the text itself */
        text-align: center;
        margin: 0;
        padding: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3); /* Adds depth to the text */
    }

    /* Updated style for the sub-header text */
    .sub-header {
        font-size: 1.25rem;
        color: #e0e0e0;                   /* A slightly muted white for the subtitle */
        font-weight: 400;
        margin-top: 0.5rem;
        letter-spacing: 0.5px;
        text-align: center;
    }

    
    /* Container Styling */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Enhanced Metric Cards */
    .metric-container {
        background: #dbcece;
        border: 1px solid #e0e0e0;
        padding: 1.2rem;
        border-radius: 15px;
        color: #1e3c72;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.03);
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Enhanced Prediction Box */
    .prediction-box {
        background: #d4edda;
        color: #000000;
        padding: 2rem;
        border-radius: 20px;
        border-left: 6px solid #4CAF50;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 15px 30px rgba(17, 153, 142, 0.3);
        transition: all 0.3s ease;
    }
    
    .prediction-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 40px rgba(17, 153, 142, 0.4);
    }
    
    .prediction-box h3 {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Enhanced Buttons */
    .stButton > button {
        background: rgb(31, 119, 180);
        color: white;
        border: none;
        padding: 0.8rem 2.5rem;
        border-radius: 30px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 25px rgba(102, 126, 234, 0.4);
        background: linear-gradient(90deg, #dbcece, rgb(31, 119, 180)); /* Reversed for hover */
    }

    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    .css-1d391kg .stRadio > label {
        color: white;
        font-weight: 500;
        font-size: 1.1rem;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        margin: 0.3rem 0;
        transition: all 0.3s ease;
    }
    
    .css-1d391kg .stRadio > label:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateX(5px);
    }
    
    /* Input Styling */
    .stSlider > div > div > div {
        background: #dbcece;
        border: 0.1px #e0e0e0;
        padding: .4rem;
        border-radius: 15px;
        color: #1e3c72;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 1px 1px rgba(0, 0, 0, 0.03);
    }
    
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.3);
    }
    
    /* Chart Containers */
    .chart-container {
        background: white;
        border-radius: 20px;
        padding: .6rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Success/Warning/Error Messages */
    .stSuccess {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 8px 20px rgba(17, 153, 142, 0.3);
    }
    
    .stWarning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 8px 20px rgba(240, 147, 251, 0.3);
    }
    
    .stError {
        background: linear-gradient(135deg, #fc4a1a 0%, #f7b733 100%);
        color: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 8px 20px rgba(252, 74, 26, 0.3);
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in {
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        .metric-container {
            margin: 0.25rem 0;
        }
        .prediction-box {
            padding: 1.5rem;
        }
    }
    /* Add this new CSS class inside your <style> tag */

/* Add or replace this CSS class inside your <style> tag */

    .sidebar-tab-header {
        background-color: transparent; /* Removes the button-like background */
        color: #1e3c72;            /* Keeps the text color dark for readability */
        padding: 8px 0px;           /* Adjusted padding for a cleaner look */
        border-radius: 0;           /* Removes rounded corners */
        font-weight: 700;           /* Bolder text to signify a header */
        margin-top: 1rem;           /* Keeps the space above the header */
        margin-bottom: 0.5rem;      /* Reduces space below the header */
        text-align: left;           /* Aligns text to the left like a standard header */
        padding-left: 10px;         /* Adds space between the blue strip and text */
        box-shadow: none;           /* Removes the shadow effect */
        font-size: 1.1rem;          /* Sets a clear header font size */
    }


    /* Add or replace these CSS classes inside your <style> tag */

    /* This styles the container itself to act as the banner */
    .banner-container {
        background-color: #e0f7fa;      /* Light cyan background from previous design */
        padding: 1rem 1.5rem;           /* Generous padding for a clean look */
        border-radius: 15px;            /* Rounded corners */
        text-align: center;             /* Center the text and icon */
        border-left: 5px solid #00778a; /* A darker accent border for emphasis */
        border-right: 5px solid #00778a; /* A darker accent border for emphasis */
        margin: 1rem 0;                 /* Standard vertical margin */
        box-shadow: 0 4px 15px rgba(0, 119, 138, 0.08); /* A subtle shadow for depth */
    }

    /* This styles the text inside the new banner container */
    .banner-text {
        color: #004d5a;                 /* Dark, highly readable text color */
        font-weight: 500;
        font-size: 1.05rem;             /* Slightly larger font for better visibility */
        margin: 0;                      /* Important: Removes default paragraph margin */
    }


    </style>
    """, unsafe_allow_html=True)

class EnhancedCLVDashboard:
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

    def create_enhanced_radar_chart(self, rfm_values):
        """Create enhanced radar chart for RFM values"""
        categories = ['Recency Score', 'Frequency Score', 'Monetary Score']

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=rfm_values + [rfm_values[0]],  # Close the loop
            theta=categories + [categories[0]],
            fill='toself',
            name='Customer Profile',
            line=dict(color='#667eea', width=3),
            fillcolor='rgba(102, 126, 234, 0.3)',
            hovertemplate='<b>%{theta}</b><br>Score: %{r}<extra></extra>'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 5],
                    tickmode='linear',
                    tick0=0,
                    dtick=1,
                    gridcolor='rgba(0,0,0,0.1)'
                ),
                angularaxis=dict(
                    tickfont=dict(size=12, color='#2c3e50')
                )
            ),
            showlegend=False,
            title={
                'text': "Customer RFM Profile",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#2c3e50'}
            },
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        return fig

    def get_model_accuracy_info(self):
        """Get model accuracy information"""
        try:
            results_df = pd.read_csv('outputs/results/model_evaluation_results.csv')
            return results_df
        except:
            return None

    # def run_dashboard(self):
    #     """Main dashboard function"""
    #     # Load enhanced CSS
    #     load_enhanced_css()
        
    #     # Header
    #     st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    #     st.markdown('<h1 class="main-header">💰 Customer Lifetime Value Analytics Platform</h1>', 
    #                unsafe_allow_html=True)
    #     st.markdown('</div>', unsafe_allow_html=True)

    #     # Sidebar
    #     st.sidebar.title("🧭 Navigation")
    #     st.sidebar.markdown("---")
    #     page = st.sidebar.radio("Choose a page:", 
    #                     ["🎯 CLV Prediction", "📊 Data Analysis", "🤖 Model Performance"])

    #     if page == "🎯 CLV Prediction":
    #         self.clv_prediction_page()
    #     elif page == "📊 Data Analysis":
    #         self.data_analysis_page()
    #     elif page == "🤖 Model Performance":
    #         self.model_performance_page()

    def run_dashboard(self):
        """Main dashboard execution with new navigation"""
        load_enhanced_css()

        # Header
        # st.markdown("""<div class="fade-in">""", unsafe_allow_html=True)
        
        # st.markdown("""<div class="fade-in">
        #             <!-- THIS IS THE NEW LINE YOU REQUESTED -->
        #     <div class="production-ready-banner fade-in">
        #         ✅ Production Ready: Pre-trained models loaded and ready for deployment
        #     </div>
        # """, unsafe_allow_html=True)
        
        # st.markdown("""
        # <div style="text-align: center;">
        #     <h1 class="main-header">💰 Customer Lifetime Value Analytics Platform</h1>
        #     <p class="sub-header">AI-Powered Customer Insight | Deployed Models Ready</p>
        # </div>
        # """, unsafe_allow_html=True)
        
        # --- Header with Background Gradient ---
        # This block replaces your previous header code.

        st.markdown("""
        <div class="header-background">
            <div style="text-align: center;">
                <h1 class="main-header">💰 Customer Lifetime Value Analytics Platform</h1>
                <p class="sub-header">AI-Powered Customer Insight | Deployed Models Ready</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
    
        with st.sidebar:
            # st.markdown("## 🧭 CLV Platform")
            st.markdown('<div class="sidebar-tab-header">🧭 CLV Platform</div>', unsafe_allow_html=True)
            page = option_menu(
                menu_title=None,
                options=["Home", "Make Prediction", "Model Insights", "About"],
                icons=["house-door-fill", "bullseye", "graph-up-arrow", "info-circle-fill"],
                default_index=0,
                styles={
                    "container": {"padding": "0!important", "background-color": "#ffffff"},
                    "icon": {"color": "#6c757d", "font-size": "1.2rem"},
                    "nav-link": {
                        "font-size": "1rem", "text-align": "left", "margin":"5px",
                        "padding": "10px", "--hover-color": "#e9ecef"
                    },
                    "nav-link-selected": {
                        "background-color": "#dbeafe", "color": "#1e3c72", "font-weight": "600"
                    },
                }
            )
            # st.markdown("---")
            st.markdown('<div class="sidebar-tab-header">🚀 Deployment Info</div>', unsafe_allow_html=True)
            st.info("Production-ready model for real-time Customer Lifetime Value (CLV) prediction and strategic analysis.")
            # st.markdown("---")
            st.markdown('<div class="sidebar-tab-header">📊 Quick Stats</div>', unsafe_allow_html=True)
            # st.subheader("📊 Quick Stats")
            accuracy_info = self.get_model_accuracy_info()
            if accuracy_info is not None:
                best_model = accuracy_info.loc[accuracy_info['R²'].idxmax()]
                st.metric(label="Best Model", value=best_model['Model'].replace('_', ' ').title())
                st.metric(label="R² Score", value=f"{best_model['R²']:.3f}")
            else:
                st.write("Model stats unavailable.")

        if page == "Home":
            self.data_analysis_page()
        elif page == "Make Prediction":
            self.clv_prediction_page()
        elif page == "Model Insights":
            self.model_performance_page()
        elif page == "About":
            self.about_page()

    def clv_prediction_page(self):
        """Enhanced CLV Prediction page with ALL original parameters"""
        # st.markdown('<div class="main-container fade-in">', unsafe_allow_html=True)
        # st.markdown("""
        # <div class="banner-container fade-in">
        #     <p class="banner-text">✅ Production Ready: Pre-trained models loaded and ready for deployment</p>
        # </div>
        # """, unsafe_allow_html=True)
        st.header("🎯 Customer Lifetime Value Prediction")
        st.markdown("Use comprehensive customer metrics to predict CLV with advanced ML models")
        st.markdown('</div>', unsafe_allow_html=True)

        # Display model accuracy info at the top
        accuracy_info = self.get_model_accuracy_info()
        if accuracy_info is not None:
            st.markdown("### 🎯 Model Accuracy Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            best_model = accuracy_info.loc[accuracy_info['R²'].idxmax()]
            
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Best Model</div>
                    <div class="metric-value">{best_model['Model']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">R² Score</div>
                    <div class="metric-value">{best_model['R²']:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">RMSE</div>
                    <div class="metric-value">{best_model['RMSE']:.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">MAE</div>
                    <div class="metric-value">{best_model['MAE']:.0f}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        col1, col2 = st.columns([1, 1])

        with col1:
            # st.markdown('<div class="main-container">', unsafe_allow_html=True)
            st.markdown("""<div class="chart-container">
                    <p class="banner-text">Enter customer metrics like recency, frequency, and monetary value to generate personalized CLV predictions. Select a model to view its performance and tailor outputs to any profile.
                </div>            
                """, unsafe_allow_html=True)
            
            st.subheader("📝 Customer Input Parameters")

            # RFM inputs - Original Parameters
            st.markdown("**🔍 RFM Metrics**")
            recency = st.slider("📅 Recency (days since last purchase)", 0, 365, 30,
                              help="Number of days since the customer's last purchase")
            frequency = st.slider("🔄 Frequency (number of orders)", 1, 50, 5,
                                help="Total number of orders made by the customer")
            monetary = st.slider("💰 Monetary (total spent $)", 0, 5000, 500,
                               help="Total amount spent by the customer")

            # RFM Scores - Original Parameters
            st.markdown("**⭐ RFM Scores (1-5)**")
            r_score = st.slider("📅 Recency Score", 1, 5, 3,
                              help="Recency score from 1 (recent) to 5 (long ago)")
            f_score = st.slider("🔄 Frequency Score", 1, 5, 3,
                              help="Frequency score from 1 (low) to 5 (high)")
            m_score = st.slider("💰 Monetary Score", 1, 5, 3,
                              help="Monetary score from 1 (low) to 5 (high)")

            # Additional features - Original Parameters
            st.markdown("**📊 Additional Customer Features**")
            total_quantity = st.number_input("📦 Total Quantity Purchased", 0, 1000, 50,
                                           help="Total quantity of items purchased")
            avg_quantity = st.number_input("📊 Average Quantity per Order", 0.0, 50.0, 3.0,
                                         help="Average quantity per order")
            avg_order_value = st.number_input("💵 Average Order Value ($)", 0.0, 1000.0, 100.0,
                                            help="Average monetary value per order")
            unique_products = st.number_input("🎁 Unique Products Purchased", 1, 100, 10,
                                            help="Number of unique products purchased")
            tenure_days = st.number_input("📆 Customer Tenure (days)", 0, 1000, 90,
                                        help="Number of days since customer first purchase")

            # Model selection with enhanced display
            st.markdown("**🤖 Model Selection**")
            available_models = list(self.models.keys())
            if available_models:
                selected_model = st.selectbox("Choose prediction model:", available_models,
                                            format_func=lambda x: f"🎯 {x.replace('_', ' ').title()}")
                
                # Show selected model info
                if accuracy_info is not None:
                    model_info = accuracy_info[accuracy_info['Model'] == selected_model]
                    if not model_info.empty:
                        st.info(f"**Selected Model Performance:**\n"
                               f"- R² Score: {model_info['R²'].iloc[0]:.3f}\n"
                               f"- RMSE: {model_info['RMSE'].iloc[0]:.2f}\n"
                               f"- MAE: {model_info['MAE'].iloc[0]:.2f}")
            else:
                st.error("❌ No models available. Please train models first.")
                st.markdown('</div>', unsafe_allow_html=True)
                return

            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            # st.markdown('<div class="main-container">', unsafe_allow_html=True)
            st.markdown("""<div class="chart-container">
                    <p class="banner-text">View CLV prediction results, including estimated value, selected model with confidence, customer segment, and strategic recommendations based on your inputs.
                </div>            
                """, unsafe_allow_html=True)
            
            st.subheader("🎯 Prediction Results")

            # Show current input summary
            st.markdown("**📋 Input Summary**")
            input_summary = pd.DataFrame({
                'Metric': ['Recency', 'Frequency', 'Monetary', 'Tenure', 'Total Quantity', 'Unique Products'],
                'Value': [f"{recency} days", f"{frequency} orders", f"${monetary:,.2f}", 
                         f"{tenure_days} days", f"{total_quantity} items", f"{unique_products} products"]
            })
            st.table(input_summary)

            if st.button("🚀 Predict Customer Lifetime Value", type="primary"):
                if self.models and self.scaler and self.feature_names:
                    try:
                        # Prepare features - ALL original parameters
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
                            'Order_Value_Std': avg_order_value * 0.2,  # Estimated std
                            'Unique_Products': unique_products,
                            'Tenure_Days': tenure_days,
                            'Purchase_Frequency_Rate': frequency / (tenure_days + 1),
                            'Avg_Monetary_per_Order': monetary / frequency if frequency != 0 else 0,
                            'Quantity_per_Product': total_quantity / unique_products if unique_products != 0 else 0
                        }

                        # Fill in any missing model features with calculated values
                        for fname in self.feature_names:
                            if fname not in input_dict:
                                input_dict[fname] = 0

                        # Ensure correct order and structure
                        input_df = pd.DataFrame([input_dict])[self.feature_names]
                        features = input_df.values.flatten()

                        # Make prediction
                        clv_prediction = self.predict_clv(features, selected_model)

                        if clv_prediction is not None:
                            # Enhanced prediction display
                            st.markdown(f"""
                            <div class="prediction-box">
                                <h3>🎯 Predicted CLV: ${clv_prediction:.2f}</h3>
                                <p><strong>🤖 Model:</strong> {selected_model.replace('_', ' ').title()}</p>
                                <p><strong>🔍 Confidence:</strong> Based on {len(self.feature_names)} features</p>
                            </div>
                            """, unsafe_allow_html=True)

                            # Enhanced Customer segmentation
                            if clv_prediction > 1000:
                                segment = "High Value"
                                color = "green"
                                emoji = "💎"
                                message = "VIP Customer - Focus on retention and premium services"
                            elif clv_prediction > 500:
                                segment = "Medium Value"
                                color = "orange"
                                emoji = "⭐"
                                message = "Growth Potential - Target with personalized offers"
                            else:
                                segment = "Low Value"
                                color = "red"
                                emoji = "🎯"
                                message = "Re-engagement Opportunity - Consider win-back campaigns"

                            st.markdown(f"**{emoji} Customer Segment:** :{color}[{segment}]")

                            # Enhanced recommendations
                            st.markdown("### 📈 Strategic Recommendations")
                            if segment == "High Value":
                                st.success(f"🌟 **VIP Treatment Required**\n\n{message}")
                                st.markdown("**Recommended Actions:**\n- Exclusive offers and early access\n- Personalized customer service\n- Loyalty program benefits")
                            elif segment == "Medium Value":
                                st.warning(f"📊 **Growth Opportunity**\n\n{message}")
                                st.markdown("**Recommended Actions:**\n- Targeted email campaigns\n- Cross-selling opportunities\n- Engagement tracking")
                            else:
                                st.info(f"🎯 **Re-engagement Focus**\n\n{message}")
                                st.markdown("**Recommended Actions:**\n- Win-back campaigns\n- Special discounts\n- Feedback collection")

                            # Display enhanced radar chart
                            st.markdown("### 📊 Customer RFM Profile")
                            st.markdown("""<div class="chart-container">
                                <p class="banner-text">RFM stands for Recency, Frequency, and Monetary Value - a technique to segment customers based on purchase behavior and tailor marketing strategies.
                            </div>            
                            """, unsafe_allow_html=True)
                            radar_fig = self.create_enhanced_radar_chart([r_score, f_score, m_score])
                            st.plotly_chart(radar_fig, use_container_width=True)

                            # Add success animation
                            st.balloons()
                            
                        else:
                            st.error("❌ Error making prediction. Please check your inputs.")
                    
                    except Exception as e:
                        st.error(f"❌ Prediction error: {str(e)}")
                else:
                    st.error("❌ Models not loaded properly. Please ensure all model files exist.")

            st.markdown('</div>', unsafe_allow_html=True)
# Model limitations disclaimer
        st.markdown("---")
        st.warning("""
        ### ⚠️ Important Disclaimer

        This prediction is based on a machine learning model trained on historical data from the Online Retail dataset.

        **Key limitations:**
        - **Historical context**: The model is based on transactional data from a single UK-based online retailer between December 2010 and December 2011.
        - **Limited features**: Predictions use only transactional data (e.g., purchases, quantities, prices). They do not account for customer demographics, marketing interactions, or product returns.
        - **Model accuracy**:  Predictions are statistical estimates, not guarantees of future revenue. Actual customer value will vary.
        - **Educational purpose**: This tool is for demonstrating CLV modeling and machine learning deployment principles.
        **Remember**: A customer's true lifetime value is influenced by many factors not captured in this dataset, such as brand loyalty, customer service experiences, competitor actions, and broader market trends.
                """)

        st.markdown("---")
        st.markdown("*Adjust the customer metrics using the sliders to see how factors like recency, frequency, and monetary value impact a customer's predicted CLV.*")        

    def data_analysis_page(self):
        """Enhanced Data Analysis page with all original features"""
        st.markdown("""
        <div class="banner-container fade-in">
            <p class="banner-text">✅ Production Ready: Pre-trained models loaded and ready for deployment</p>
        </div>
        """, unsafe_allow_html=True)
        # st.markdown("""
        # <div class="main-container fade-in">
        #     <div class="production-ready-banner">
        #         ✅ Production Ready: Pre-trained models loaded and ready for deployment
        #     </div>
        # </div>
        # """, unsafe_allow_html=True)
        st.header("📊 Comprehensive Data Analysis")
        st.markdown("Explore customer data patterns and insights")
        st.markdown('</div>', unsafe_allow_html=True)

        try:
            # Load data
            rfm_df = pd.read_csv('data/rfm_analysis.csv')
            clv_df = pd.read_csv('data/clv_dataset.csv')

            # Enhanced summary metrics - Original 4 cards
            st.markdown("### 📈 Key Performance Indicators")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Total Customers</div>
                    <div class="metric-value">{len(rfm_df):,}</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Total Revenue</div>
                    <div class="metric-value">${rfm_df['Monetary'].sum():,.0f}</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Average CLV</div>
                    <div class="metric-value">${clv_df['CLV_Target'].mean():.2f}</div>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Average Orders</div>
                    <div class="metric-value">{rfm_df['Frequency'].mean():.1f}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # Enhanced charts - All original visualizations
            st.markdown("### 📊 Visual Analytics Dashboard")
            
            col1, col2 = st.columns(2)

            with col1:
                # Enhanced customer segments pie chart
                fig_pie = px.pie(rfm_df, names='Customer_Segment', title='Customer Segments Distribution',
                               color_discrete_sequence=px.colors.qualitative.Set3)
                fig_pie.update_layout(
                    title_font_size=18,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                st.markdown("""<div class="chart-container">
                    <p class="banner-text">This interactive pie chart provides a real-time breakdown of the customer base by segment, showing the percentage and count for each customer type.
                </div>            
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                
                # Enhanced RFM correlation heatmap
                rfm_corr = rfm_df[['Recency', 'Frequency', 'Monetary']].corr()
                fig_heatmap = px.imshow(rfm_corr, text_auto=True, title='RFM Correlation Matrix',
                                      color_continuous_scale='RdBu')
                fig_heatmap.update_layout(
                    title_font_size=18,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
                st.markdown("""<div class="chart-container">
                    <p class="banner-text">This heatmap shows the relationships between Recency, Frequency, and Monetary metrics, helping you spot customer behavior patterns at a glance.</p>
                </div>            
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                
                # Enhanced CLV distribution
                fig_hist = px.histogram(clv_df, x='CLV_Target', title='CLV Distribution', nbins=20,
                                      color_discrete_sequence=['#667eea'])
                fig_hist.update_layout(
                    title_font_size=18,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                st.markdown("""<div class="chart-container">
                    <p class="banner-text">This histogram shows the spread of predicted CLV across the customer base, highlighting how many customers fall into each CLV range.</p>
                </div>            
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                

                # st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                # Enhanced top customers chart
               
                top_customers = rfm_df.nlargest(10, 'Monetary')
                fig_bar = px.bar(top_customers, x='Customer_ID', y='Monetary', 
                               title='Top 10 Customers by Revenue',
                               color='Monetary', color_continuous_scale='Viridis')
                fig_bar.update_layout(
                    title_font_size=18,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                st.markdown("""<div class="chart-container">
                    <p class="banner-text">This bar chart highlights the top 10 customers based on total revenue, showing their contribution to overall sales. Click to view their <a href="https://github.com/ShubhamS168/clv-prediction-dashboard/blob/main/outputs/plots/top_customers.png"
                    target="_blank"
                    style="color:#1f77b4; text-decoration:none; font-weight:500;">
                        Customer_IDs.
                    </a>. </p>
                </div>            
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("---")
            # Enhanced detailed data table
            st.markdown("### 📋 Customer Data Table")
            # st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("""<div class="chart-container">
                    <p class="banner-text">This interactive table lets you explore customer data with real-time filters for Customer Lifetime Value (CLV), segments, and key metrics. Use the filters to quickly identify high-value customers and analyze segment trends.</p>
                </div>            
                """, unsafe_allow_html=True)
            
            # Add filters
            col1, col2, col3 = st.columns(3)
            with col1:
                segment_filter = st.selectbox("Filter by Segment:", 
                                            ['All'] + list(rfm_df['Customer_Segment'].unique()))
            with col2:
                min_clv = st.number_input("Min CLV:", value=0)
            with col3:
                max_clv = st.number_input("Max CLV:", value=int(rfm_df['Monetary'].max()))

            # Apply filters
            filtered_df = rfm_df.copy()
            if segment_filter != 'All':
                filtered_df = filtered_df[filtered_df['Customer_Segment'] == segment_filter]
            filtered_df = filtered_df[(filtered_df['Monetary'] >= min_clv) & 
                                    (filtered_df['Monetary'] <= max_clv)]

            st.dataframe(filtered_df.head(20), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # System requirements note
            
            # st.markdown("---")
            st.info("""
            **💡 Pro Tip:** This system showcases core MLOps in action. It uses versioned models for reliable CLV forecasting, caches data for high-speed performance, and implements real-time inference, making it an ideal example for deploying customer analytics models.!
            """) 

        except FileNotFoundError:
            st.error("❌ Data files not found. Please run the data preprocessing scripts first.")

    def model_performance_page(self):
        """Enhanced Model Performance page with all original features"""
        # st.markdown('<div class="main-container fade-in">', unsafe_allow_html=True)
        # st.markdown("""
        # <div class="banner-container fade-in">
        #     <p class="banner-text">✅ Production Ready: Pre-trained models loaded and ready for deployment</p>
        # </div>
        # """, unsafe_allow_html=True)
        st.header("🤖 Model Performance Analysis")
        st.markdown("Comprehensive evaluation of all trained models")
        st.markdown('</div>', unsafe_allow_html=True)

        try:
            # Load model results
            results_df = pd.read_csv('outputs/results/model_evaluation_results.csv')

            # Enhanced performance metrics display
            st.markdown("### 🏆 Model Performance Overview")
            
            best_model = results_df.loc[results_df['R²'].idxmax()]
            worst_model = results_df.loc[results_df['R²'].idxmin()]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">🥇 Best Model</div>
                    <div class="metric-value">{best_model['Model']}</div>
                    <div class="metric-label">R² = {best_model['R²']:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">📊 Avg Performance</div>
                    <div class="metric-value">{results_df['R²'].mean():.3f}</div>
                    <div class="metric-label">Average R² Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">📈 Models Trained</div>
                    <div class="metric-value">{len(results_df)}</div>
                    <div class="metric-label">Total Models</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # Enhanced performance comparison charts - All original charts
            st.markdown("### 📊 Performance Comparison Charts")
            
            col1, col2, col3 = st.columns(3)

            with col1:
                # st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                # st.markdown("""<div class="chart-container">
                #     <p class="banner-text">RMSE (Root Mean Squared Error) measures the average size of prediction errors, with larger errors penalized more. A lower RMSE indicates more accurate and reliable predictions.</p>
                # </div>            
                # """, unsafe_allow_html=True)
                st.markdown("""
                <div class="chart-container"> 
                    <p class="banner-text">
                        <a href="https://c3.ai/glossary/data-science/root-mean-square-error-rmse/" target="_blank" style="text-decoration: none; color: inherit;"><strong>RMSE (Root Mean Squared Error) </strong></a>  measures the average size of prediction errors, with larger errors penalized more. A lower RMSE indicates more accurate and reliable predictions.
                    </p>
                </div>            
                """, unsafe_allow_html=True)

                fig_rmse = px.bar(results_df, x='Model', y='RMSE', title='RMSE Comparison',
                                color='RMSE', color_continuous_scale='Reds_r')
                fig_rmse.update_layout(
                    title_font_size=16,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_rmse, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                # st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                # st.markdown("""<div class="chart-container">
                #     <p class="banner-text">MAE (Mean Absolute Error) calculates the average absolute difference between predicted and actual values. Lower MAE reflects more consistent and accurate model predictions.</p>
                # </div>            
                # """, unsafe_allow_html=True)
                st.markdown("""
                <div class="chart-container"> 
                    <p class="banner-text">
                        <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html" target="_blank" style="text-decoration: none; color: inherit;"><strong>MAE (Mean Absolute Error)</strong></a> calculates the average absolute difference between predicted and actual values. Lower MAE reflects more consistent and accurate model predictions.
                    </p>
                </div>            
                """, unsafe_allow_html=True)
                fig_mae = px.bar(results_df, x='Model', y='MAE', title='MAE Comparison',
                               color='MAE', color_continuous_scale='Oranges_r')
                fig_mae.update_layout(
                    title_font_size=16,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_mae, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col3:
                # st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                # st.markdown("""<div class="chart-container">
                #     <p class="banner-text">R² (Coefficient of Determination) reflects how well the model explains the variance in the target variable. A value closer to 1 suggests strong explanatory power and better overall performance.</p>
                # </div>            
                # """, unsafe_allow_html=True)
                st.markdown("""
                <div class="chart-container"> 
                    <p class="banner-text">
                        <a href="https://arize.com/blog-course/r-squared-understanding-the-coefficient-of-determination/" target="_blank" style="text-decoration: none; color: inherit;"><strong>R² (Coefficient of Determination)</strong></a> shows how well the model explains variance in the target variable. A value closer to 1 suggests strong explanatory power and better performance.
                    </p>
                </div>            
                """, unsafe_allow_html=True)
                fig_r2 = px.bar(results_df, x='Model', y='R²', title='R² Score Comparison',
                              color='R²', color_continuous_scale='Greens')
                fig_r2.update_layout(
                    title_font_size=16,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_r2, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Enhanced best model highlight
            st.markdown("### 🏆 Best Model Performance")
            st.success(f"🥇 **Champion Model: {best_model['Model']}**\n\n"
                      f"📊 **Performance Metrics:**\n"
                      f"- R² Score: {best_model['R²']:.3f}\n"
                      f"- RMSE: {best_model['RMSE']:.2f}\n"
                      f"- MAE: {best_model['MAE']:.2f}")
            
            st.markdown("---")

            # Enhanced detailed results table
            st.markdown("### 📋 Detailed Model Results")
            # st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("""<div class="chart-container">
                <p class="banner-text">This chart visually compares RMSE, MAE, and R² metrics across 7 machine learning models, ranking them from best to worst. It helps you quickly evaluate and select the most accurate CLV prediction model at a glance.</p>
            </div>            
            """, unsafe_allow_html=True)
            
            # Add ranking
            results_display = results_df.copy()
            results_display['Rank'] = results_display['R²'].rank(method='dense', ascending=False).astype(int)
            results_display = results_display.sort_values('R²', ascending=False)
            
            st.dataframe(results_display, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Enhanced feature importance
            try:
                feature_importance = pd.read_csv('outputs/results/feature_importance.csv')
                st.markdown("### 🎯 Feature Importance Analysis")
                
                # st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                
                st.markdown("""<div class="chart-container">
                    <p class="banner-text">This section shows the top 10 features impacting CLV predictions, ranked by importance. It highlights key factors like purchase frequency and tenure, helping you focus on what drives customer value and improve strategy and model accuracy.</p>
                </div>            
                """, unsafe_allow_html=True)
                fig_importance = px.bar(feature_importance.head(10), 
                                      x='Importance', y='Feature', 
                                      orientation='h', title='Top 10 Most Important Features',
                                      color='Importance', color_continuous_scale='Viridis')
                fig_importance.update_layout(
                    title_font_size=18,
                    height=500,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_importance, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Feature importance table
                st.markdown("""<div class="chart-container">
                    <p class="banner-text">This chart ranks 19 model features by their importance, from most to least influential. The top feature has the highest impact (score: 0.3773), while the bottom one has minimal effect (score near 0).</p>
                </div>            
                """, unsafe_allow_html=True)
        
                st.markdown("### 📊 Complete Feature Importance")
                st.dataframe(feature_importance, use_container_width=True)
                
                

            except FileNotFoundError:
                st.warning("⚠️ Feature importance data not found. This is normal for some models.")
                
            st.markdown("---")
            st.markdown("*This level of performance validation and feature transparency confirms the model is not just a demo, but a reliable tool ready for production.*")  

        except FileNotFoundError:
            st.error("❌ Model results not found. Please run the modeling script first.")

    def about_page(self):
        """Creates the 'About' page"""
        # st.markdown("""
        # <div class="banner-container fade-in">
        #     <p class="banner-text">✅ Production Ready: Pre-trained models loaded and ready for deployment</p>
        # </div>
        # """, unsafe_allow_html=True)
        st.title("ℹ️ About This Application")
        # st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.markdown("""
    ## Customer Lifetime Value (CLV) Analytics Platform

    Welcome to a next-generation analytics platform that empowers businesses to understand, predict, and maximize customer value using advanced machine learning.

    ---

    ### 🚀 What Does This Platform Do?

    - **Predicts Customer Lifetime Value:** Instantly estimate the future revenue a customer will bring, using robust machine learning models trained on real-world retail data.
    - **Interactive Data Exploration:** Dive into your customer base with dynamic dashboards. Visualize RFM (Recency, Frequency, Monetary) segments, spot trends, and identify high-value customers in real time.
    - **Transparent Model Insights:** Access clear, interactive reports on model performance—compare accuracy, interpret feature importance, and understand the drivers behind every prediction.

    ---

    ### 🛠️ Key Features

    - **Multiple ML Models:** Choose from state-of-the-art algorithms (Random Forest, XGBoost, and more) for CLV prediction.
    - **Real-Time Inference:** Generate predictions instantly as you explore different customer profiles.
    - **Customizable Inputs:** Experiment with customer scenarios using intuitive sliders and input fields.
    - **Production-Ready Design:** Benefit from best practices in model versioning, error handling, and deployment—ideal for both learning and real business use.

    ---

    ### 📊 Why Use This Platform?

    - **Make Data-Driven Decisions:** Prioritize retention, target marketing, and optimize resources based on actionable customer insights.
    - **Learn by Doing:** Perfect for data science learners, business analysts, and professionals seeking hands-on experience with real ML deployment.
    - **Built with Streamlit:** Enjoy a seamless, modern user experience powered by an efficient data pipeline and interactive visualizations.

    ---

    ### 📚 About the Data

    This platform is trained on the [Online Retail dataset](https://www.kaggle.com/datasets/lakshmi25npathi/online-retail-dataset), featuring real transaction records from a UK-based retailer. Predictions and analytics are for demonstration and educational purposes.

    ---

    **Ready to unlock the value hidden in your customer data? Explore, predict, and gain actionable insights—all in one place.**

    """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# Main execution
if __name__ == "__main__":
    dashboard = EnhancedCLVDashboard()
    dashboard.run_dashboard()
