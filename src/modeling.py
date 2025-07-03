
"""
Modeling Module for CLV Prediction Project
This module handles machine learning model training and evaluation for CLV prediction
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import XGBoost, handle if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

class CLVModelTrainer:
    """
    Class to handle CLV model training and evaluation
    """

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.results = {}

    # def prepare_data(self, df, target_column='CLV_Target', test_size=0.2, random_state=42):
    #     """
    #     Prepare data for modeling

    #     Parameters:
    #     df (pd.DataFrame): Dataset with features and target
    #     target_column (str): Name of target column
    #     test_size (float): Proportion of test set
    #     random_state (int): Random seed

    #     Returns:
    #     tuple: X_train, X_test, y_train, y_test
    #     """
    #     # Select feature columns (exclude ID and target)
    #     feature_cols = [col for col in df.columns if col not in ['Customer_ID', target_column, 'Customer_Segment']]

    #     X = df[feature_cols]
    #     y = df[target_column]

    #     self.feature_names = feature_cols

    #     # Split data
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X, y, test_size=test_size, random_state=random_state
    #     )

    #     # Scale features
    #     X_train_scaled = self.scaler.fit_transform(X_train)
    #     X_test_scaled = self.scaler.transform(X_test)

    #     print(f"Training set size: {X_train.shape}")
    #     print(f"Test set size: {X_test.shape}")
    #     print(f"Features: {len(feature_cols)}")

    #     return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test
    
    def prepare_data(self, df, target_column='CLV_Target', test_size=0.2, random_state=42):
        '''
        Prepare data for modeling

        Parameters:
        df (pd.DataFrame): Dataset with features and target
        target_column (str): Name of target column
        test_size (float): Proportion of test set
        random_state (int): Random seed

        Returns:
        tuple: X_train, X_test, y_train, y_test, X_train_original, X_test_original
        '''
        # Select feature columns (exclude ID and target)
        feature_cols = [col for col in df.columns if col not in ['Customer_ID', target_column, 'Customer_Segment']]

        X = df[feature_cols].copy()
        y = df[target_column]

        # Handle datetime columns
        datetime_cols = X.select_dtypes(include=['datetime64', 'datetime64[ns]', 'object']).columns
        for col in datetime_cols:
            if pd.api.types.is_datetime64_any_dtype(X[col]) or 'date' in col.lower():
                try:
                    X[col] = pd.to_datetime(X[col], errors='coerce')
                    X[col + '_year'] = X[col].dt.year
                    X[col + '_month'] = X[col].dt.month
                    X[col + '_day'] = X[col].dt.day
                    X.drop(columns=[col], inplace=True)
                except Exception as e:
                    print(f"Warning: Could not convert column {col} to datetime: {e}")
                    X.drop(columns=[col], inplace=True)

        self.feature_names = X.columns.tolist()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"Training set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        print(f"Features: {len(self.feature_names)}")

        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test


    def train_models(self, X_train, y_train):
        """
        Train multiple models for CLV prediction

        Parameters:
        X_train: Training features
        y_train: Training target
        """
        print("Training models...")

        # Define models
        models_config = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0)
        }

        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models_config['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=42)

        # Train models
        for name, model in models_config.items():
            print(f"Training {name}...")
            try:
                model.fit(X_train, y_train)
                self.models[name] = model
                print(f"✓ {name} trained successfully")
            except Exception as e:
                print(f"✗ Error training {name}: {e}")

    def evaluate_models(self, X_test, y_test):
        """
        Evaluate trained models

        Parameters:
        X_test: Test features
        y_test: Test target

        Returns:
        pd.DataFrame: Evaluation results
        """
        print("\nEvaluating models...")

        results = []

        for name, model in self.models.items():
            try:
                # Make predictions
                y_pred = model.predict(X_test)

                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                results.append({
                    'Model': name,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R²': r2
                })

                print(f"{name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

            except Exception as e:
                print(f"Error evaluating {name}: {e}")

        self.results = pd.DataFrame(results)
        return self.results

    def get_feature_importance(self, model_name='Random Forest', X_train_original=None):
        """
        Get feature importance from tree-based models

        Parameters:
        model_name (str): Name of the model
        X_train_original: Original training features (unscaled) for column names

        Returns:
        pd.DataFrame: Feature importance
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return None

        model = self.models[model_name]

        # Check if model has feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

            return importance_df
        else:
            print(f"Model {model_name} does not have feature importance")
            return None

    def predict_clv(self, customer_features, model_name='Random Forest'):
        """
        Predict CLV for new customers

        Parameters:
        customer_features: Features for prediction
        model_name (str): Model to use for prediction

        Returns:
        array: CLV predictions
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return None

        # Scale features
        if len(customer_features.shape) == 1:
            customer_features = customer_features.reshape(1, -1)

        customer_features_scaled = self.scaler.transform(customer_features)

        # Make prediction
        prediction = self.models[model_name].predict(customer_features_scaled)

        return prediction

    def save_models(self, save_path='../outputs/models/'):
        """
        Save trained models and scaler

        Parameters:
        save_path (str): Path to save models
        """
        import os
        os.makedirs(save_path, exist_ok=True)

        # Save models
        for name, model in self.models.items():
            filename = f"{save_path}clv_model_{name.lower().replace(' ', '_')}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {name} model to {filename}")

        # Save scaler
        scaler_path = f"{save_path}scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Saved scaler to {scaler_path}")

        # Save feature names
        features_path = f"{save_path}feature_names.pkl"
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_names, f)
        print(f"Saved feature names to {features_path}")

def hyperparameter_tuning(X_train, y_train, model_type='RandomForest'):
    """
    Perform hyperparameter tuning for selected model

    Parameters:
    X_train: Training features
    y_train: Training target
    model_type (str): Type of model to tune

    Returns:
    Best model after tuning
    """
    print(f"Performing hyperparameter tuning for {model_type}...")

    if model_type == 'RandomForest':
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_type == 'XGBoost' and XGBOOST_AVAILABLE:
        model = xgb.XGBRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
    else:
        print(f"Hyperparameter tuning not configured for {model_type}")
        return None

    # Perform grid search
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {-grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('../data/clv_dataset.csv')

    # Initialize trainer
    trainer = CLVModelTrainer()

    # Prepare data
    X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = trainer.prepare_data(df)

    # Train models
    trainer.train_models(X_train, y_train)

    # Evaluate models
    results = trainer.evaluate_models(X_test, y_test)

    # Display results
    print("\n=== Model Evaluation Results ===")
    print(results.to_string(index=False))

    # Get feature importance
    feature_importance = trainer.get_feature_importance('Random Forest')
    if feature_importance is not None:
        print("\n=== Feature Importance (Random Forest) ===")
        print(feature_importance.head(10).to_string(index=False))

    # Save models
    trainer.save_models()

    # Save results
    results.to_csv('../outputs/model_evaluation_results.csv', index=False)
    if feature_importance is not None:
        feature_importance.to_csv('../outputs/feature_importance.csv', index=False)

    print("\nModeling completed! Results saved to outputs/ folder.")
