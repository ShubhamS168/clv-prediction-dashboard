# Customer Lifetime Value (CLV) Prediction Project

🚀 A comprehensive machine learning project for predicting Customer Lifetime Value using the Online Retail Dataset from Kaggle.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Models](#models)
- [Results](#results)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This project implements a complete CLV prediction pipeline including:
- **Data cleaning and preprocessing**
- **RFM (Recency, Frequency, Monetary) analysis**
- **Multiple machine learning models**
- **Interactive Streamlit dashboard**
- **Comprehensive visualizations**
- **Model evaluation and comparison**

## 📊 Dataset

The project uses the **Online Retail Dataset** from Kaggle:
- **Source**: https://www.kaggle.com/datasets/lakshmi25npathi/online-retail-dataset
- **Description**: E-commerce transactions from a UK-based retailer
- **Time Period**: December 2009 - December 2011
- **Features**: Invoice, StockCode, Description, Quantity, InvoiceDate, Price, CustomerID, Country

## 📁 Project Structure

```
clv-prediction-project/
├── data/                          # Data files
│   ├── online_retail_dataset.csv  # Original dataset
│   ├── cleaned_retail_data.csv    # Cleaned data
│   ├── rfm_analysis.csv          # RFM metrics
│   └── clv_dataset.csv           # Final modeling dataset
├── src/                          # Python modules
│   ├── data_cleaning.py          # Data preprocessing
│   ├── feature_engineering.py    # RFM analysis and feature creation
│   ├── modeling.py               # ML model training and evaluation
│   └── visualization.py          # Plotting and visualization
├── notebooks/                    # Jupyter notebooks
│   └── CLV_Modeling.ipynb        # Main analysis notebook
├── outputs/                      # Generated outputs
│   ├── plots/                    # Visualization files
│   └── models/                   # Trained models and scalers
├── streamlit_app.py              # Interactive dashboard
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/clv-prediction-project.git
cd clv-prediction-project
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download the dataset:**
   - Download from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/online-retail-dataset)
   - Place `online_retail_dataset.csv` in the `data/` folder

## 🚀 Usage

### Option 1: Run the Complete Pipeline

```bash
# 1. Clean the data
cd src
python data_cleaning.py

# 2. Perform feature engineering
python feature_engineering.py

# 3. Train models
python modeling.py

# 4. View results in the notebook
jupyter notebook ../notebooks/CLV_Modeling.ipynb
```

### Option 2: Use the Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

## ✨ Features

### Data Processing
- ✅ Remove missing CustomerID records
- ✅ Filter out canceled orders
- ✅ Handle negative quantities and prices
- ✅ Create derived features (TotalPrice)
- ✅ Outlier detection and removal

### RFM Analysis
- ✅ **Recency**: Days since last purchase
- ✅ **Frequency**: Number of orders
- ✅ **Monetary**: Total amount spent
- ✅ Customer segmentation based on RFM scores
- ✅ Additional features (tenure, purchase rate, etc.)

### Machine Learning Models
- ✅ Linear Regression
- ✅ Ridge Regression
- ✅ Lasso Regression
- ✅ Random Forest
- ✅ Gradient Boosting
- ✅ Support Vector Regression
- ✅ XGBoost (optional)

### Visualizations
- ✅ RFM distribution plots
- ✅ Customer segmentation charts
- ✅ Model performance comparison
- ✅ Feature importance analysis
- ✅ Actual vs Predicted plots
- ✅ Comprehensive dashboard

## 🤖 Models

The project implements multiple regression models for CLV prediction:

| Model | Description | Use Case |
|-------|-------------|----------|
| **Linear Regression** | Simple baseline model | Quick insights |
| **Random Forest** | Ensemble method with feature importance | Best overall performance |
| **XGBoost** | Gradient boosting | High accuracy |
| **Ridge/Lasso** | Regularized linear models | Feature selection |

### Model Evaluation Metrics
- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **R²** (R-squared Score)

## 📈 Results

### Sample Results (will vary based on your data):
- **Best Model**: Random Forest
- **R² Score**: ~0.85
- **RMSE**: ~$50-100
- **Key Features**: Monetary, Frequency, Recency

### Customer Segments
- **Champions**: High value, frequent buyers
- **Loyal Customers**: Regular, valuable customers
- **At Risk**: Previously valuable, now inactive
- **Lost**: Haven't purchased recently

## 🖥️ Streamlit Dashboard

The interactive dashboard provides:

### 🎯 CLV Prediction Page
- Input customer RFM metrics
- Select prediction model
- Get CLV prediction and customer segment
- Visual RFM profile (radar chart)
- Personalized recommendations

### 📊 Data Analysis Page
- Dataset summary statistics
- Customer segment distribution
- Revenue analysis
- Top customers ranking
- Interactive data tables

### 🤖 Model Performance Page
- Model comparison charts
- Performance metrics
- Feature importance visualization
- Best model identification

### Dashboard Screenshots
*Add screenshots of your dashboard here*

## 📊 Sample Predictions

```python
# Example: Predict CLV for a customer
customer_features = {
    'Recency': 30,      # Days since last purchase
    'Frequency': 5,     # Number of orders
    'Monetary': 500,    # Total spent
    'R_Score': 4,       # Recency score (1-5)
    'F_Score': 3,       # Frequency score (1-5) 
    'M_Score': 4        # Monetary score (1-5)
}

predicted_clv = model.predict(customer_features)
# Output: $247.58
```

## 🛠️ Advanced Features

### Optional: BG/NBD + Gamma-Gamma Model
The project includes optional advanced CLV modeling using the `lifetimes` package:

```bash
pip install lifetimes
```

This implements:
- **BG/NBD Model**: Predicts customer purchase frequency
- **Gamma-Gamma Model**: Predicts monetary value
- **Combined CLV**: Expected revenue over time period

## 📝 Model Interpretation

### Feature Importance (Random Forest)
1. **Monetary**: Total amount spent (most important)
2. **Frequency**: Number of orders
3. **Recency**: Days since last purchase
4. **Average Order Value**: Spending per transaction
5. **Unique Products**: Product diversity

### Business Insights
- **High CLV Indicators**: Recent purchases, frequent orders, high spending
- **Risk Factors**: Long recency, low frequency, declining monetary value
- **Segmentation**: Clear customer tiers for targeted marketing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset Source**: [Kaggle Online Retail Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/online-retail-dataset)
- **RFM Analysis**: Based on marketing analytics best practices
- **Machine Learning**: Scikit-learn, XGBoost communities
- **Visualization**: Matplotlib, Seaborn, Plotly libraries

## 📞 Contact

- **Author**: [Shubham Sourav]((https://github.com/ShubhamS168))
- **Email**: shubhamsourav475@gmail.com
- **LinkedIn**: [in](www.linkedin.com/in/shubham-sourav-460493264)
- **GitHub**: [ShubhamS168](https://github.com/ShubhamS168)

---

⭐ **Star this repository if you found it helpful!** ⭐

## 🚀 Quick Start Commands

```bash
# Clone and setup
git clone https://github.com/ShubhamS168/clv-prediction-project.git
cd clv-prediction-project
pip install -r requirements.txt

# Run the pipeline
cd src && python data_cleaning.py && python feature_engineering.py && python modeling.py

# Launch dashboard  
streamlit run streamlit_app.py
```

Happy predicting! 🎉
