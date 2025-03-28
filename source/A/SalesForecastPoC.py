#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sales Forecasting Proof-of-Concept (PoC) Script
This single script is structured into logical sections to simulate a modular design.
Sections include: Configuration, Data Loading & Validation, Feature Engineering,
Model Training & Evaluation, Model Persistence, and FastAPI Serving.
"""

# =============================================================================
# MODULE: Imports and Logging Configuration
# =============================================================================
import logging
import pandas as pd
import joblib
import yaml
from typing import List, Optional
from datetime import datetime

# Configure logging for the script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("SalesForecastingPoC")

# =============================================================================
# MODULE: Configuration (YAML loading + Pydantic validation)
# =============================================================================
from pydantic import BaseModel, Field

# Define a Pydantic model for configuration settings
class Config(BaseModel):
    data_path: str = Field(..., description="Path to the input sales data CSV file")
    model_path: str = Field(..., description="Path where the trained model will be saved (joblib file)")
    lag_days: List[int] = Field(default_factory=lambda: [1, 7], description="List of lag days to use for features")
    rolling_windows: List[int] = Field(default_factory=lambda: [7], description="List of window sizes for rolling mean features")
    test_size: float = Field(0.2, description="Proportion of data to reserve for testing (0 < test_size < 1)")
    target_col: str = Field("sales", description="Name of the target column in the dataset")
    date_col: str = Field("date", description="Name of the date column in the dataset")
    id_col: str = Field("store_id", description="Name of the column representing store/item ID")
    xgb_params: dict = Field(default_factory=lambda: {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
                             description="Dictionary of parameters for XGBoost model")
    baseline_method: str = Field("last_value", description="Method for baseline forecast (e.g., 'last_value' or 'mean')")

# Load configuration from YAML file
config = None
try:
    with open("config.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    config = Config(**config_dict)
    logger.info("Configuration loaded successfully.")
except FileNotFoundError:
    logger.error("Configuration file 'config.yaml' not found.")
    raise
except Exception as e:
    logger.error("Error loading configuration: %s", e)
    raise

# For clarity, log the configuration (excluding any sensitive info if present)
logger.info("Config settings: %s", config.dict())

# Example (expected format) of config.yaml for reference:
# data_path: "data/sales_data.csv"
# model_path: "models/sales_forecast_model.joblib"
# lag_days: [1, 7]
# rolling_windows: [7]
# test_size: 0.2
# target_col: "sales"
# date_col: "date"
# id_col: "store_id"
# xgb_params: {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1}
# baseline_method: "last_value"

# Determine feature columns based on config (to use consistently in training and prediction)
feature_columns = [config.id_col] \
    + [f"lag_{lag}" for lag in config.lag_days] \
    + [f"rolling_mean_{w}" for w in config.rolling_windows] \
    + ["day_of_week", "month"]

# =============================================================================
# MODULE: Data Loading and Validation (using Pandas and Pandera)
# =============================================================================
import pandera as pa
from pandera import Column, DataFrameSchema, Check

df: Optional[pd.DataFrame] = None
try:
    # Load data into a pandas DataFrame (parse dates for date column)
    df = pd.read_csv(config.data_path, parse_dates=[config.date_col])
    logger.info("Data loaded from %s. Shape: %s", config.data_path, df.shape)
except Exception as e:
    logger.error("Failed to load data: %s", e)
    raise

# Define a schema for data validation
schema = DataFrameSchema({
    config.id_col: Column(int, nullable=False),
    config.date_col: Column(object, nullable=False),  # will be datetime after parse_dates
    config.target_col: Column(float, Check.ge(0), nullable=False)
})
try:
    df = schema.validate(df)
    logger.info("Data validation passed.")
except Exception as e:
    logger.error("Data validation error: %s", e)
    raise

# =============================================================================
# MODULE: Feature Engineering (lag features, rolling stats, date features)
# =============================================================================
def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag features, rolling statistics, and date-based features for the sales data.
    This function assumes the input DataFrame has at least the columns: id_col, date_col, target_col.
    It returns a new DataFrame with additional feature columns and drops rows with NaNs due to lagging.
    """
    df_sorted = data.sort_values([config.id_col, config.date_col])
    # Create lag features for each lag in config.lag_days
    for lag in config.lag_days:
        df_sorted[f"lag_{lag}"] = df_sorted.groupby(config.id_col)[config.target_col].shift(lag)
    # Create rolling mean features for each window in config.rolling_windows
    for window in config.rolling_windows:
        # Use shift(1) to exclude current day from the rolling window
        df_sorted[f"rolling_mean_{window}"] = df_sorted.groupby(config.id_col)[config.target_col]\
            .shift(1).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    # Date-based temporal features
    df_sorted["day_of_week"] = df_sorted[config.date_col].dt.dayofweek  # Monday=0, Sunday=6
    df_sorted["month"] = df_sorted[config.date_col].dt.month
    # Remove any rows with NaN values (from lag features at start of series)
    df_feat = df_sorted.dropna().reset_index(drop=True)
    return df_feat

# Perform feature engineering
try:
    df = create_features(df)
    logger.info("Feature engineering completed. New shape: %s", df.shape)
    logger.info("Features added: %s", [col for col in df.columns if col not in [config.id_col, config.date_col, config.target_col]])
except Exception as e:
    logger.error("Feature engineering failed: %s", e)
    raise

# =============================================================================
# MODULE: Model Training and Evaluation (Scikit-learn + XGBoost)
# =============================================================================
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

model = None  # global placeholder for the trained model

if __name__ == "__main__":
    try:
        # Separate features (X) and target (y)
        target = config.target_col
        X = df[feature_columns]
        y = df[target]
        # Use time-based split: last portion as test set (shuffle=False to maintain chronological order)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.test_size, shuffle=False)
        logger.info("Split data into train and test sets. Train size: %d, Test size: %d", len(X_train), len(X_test))
        # Baseline model prediction: use last known value (lag_1) as forecast
        if "lag_1" in X_test.columns and config.baseline_method == "last_value":
            baseline_preds = X_test["lag_1"]
        else:
            # If lag_1 isn't available or different baseline chosen, use simple mean of training target as baseline
            baseline_preds = pd.Series(y_train.mean(), index=y_test.index)
        # Calculate baseline performance
        baseline_mae = mean_absolute_error(y_test, baseline_preds)
        baseline_rmse = mean_squared_error(y_test, baseline_preds, squared=False)
        # Train XGBoost model
        model = xgb.XGBRegressor(**config.xgb_params)
        model.fit(X_train, y_train)
        logger.info("Model training completed with XGBoost.")
        # Evaluate the model on test set
        y_pred = model.predict(X_test)
        model_mae = mean_absolute_error(y_test, y_pred)
        model_rmse = mean_squared_error(y_test, y_pred, squared=False)
        logger.info("Baseline MAE: %.4f, RMSE: %.4f", baseline_mae, baseline_rmse)
        logger.info("Model   MAE: %.4f, RMSE: %.4f", model_mae, model_rmse)
        if model_mae <= baseline_mae:
            logger.info("Model outperforms baseline (MAE improvement: %.4f)", baseline_mae - model_mae)
        else:
            logger.info("Model does NOT outperform baseline (MAE difference: %.4f)", model_mae - baseline_mae)
    except Exception as e:
        logger.exception("Error during model training or evaluation: %s", e)
        raise
    # =============================================================================
    # MODULE: Model Persistence (saving trained model with joblib)
    # =============================================================================
    try:
        joblib.dump(model, config.model_path)
        logger.info("Trained model saved to %s", config.model_path)
    except Exception as e:
        logger.error("Failed to save model: %s", e)
        raise
    # After training and saving, prepare for inference usage (reload model to ensure persistence works)
    try:
        model = joblib.load(config.model_path)
        logger.info("Model loaded back from disk for verification.")
    except Exception as e:
        logger.error("Failed to load saved model: %s", e)
        raise
    # Perform a sample prediction using the trained model (for demonstration/logging)
    try:
        if not X_test.empty:
            sample_features = X_test.iloc[0:1]  # take first sample from test set
            true_value = float(y_test.iloc[0])
            pred_value = float(model.predict(sample_features)[0])
            logger.info("Sample prediction - Features: %s", sample_features.to_dict(orient="records")[0])
            logger.info("Sample prediction - Actual value: %.2f, Predicted value: %.2f", true_value, pred_value)
    except Exception as e:
        logger.error("Error during sample prediction: %s", e)
    # Instructions for running the API after training
    logger.info("Training pipeline completed. You can now run the FastAPI server with: uvicorn sales_forecasting_poc:app --reload")

# =============================================================================
# MODULE: FastAPI Model Serving (API server with /predict endpoint)
# =============================================================================
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel as PydanticBaseModel

app = FastAPI(title="Sales Forecasting API", description="A FastAPI for sales prediction", version="1.0")

# Define Pydantic models for request and response
class PredictionRequest(PydanticBaseModel):
    store_id: int
    date: datetime
    recent_sales: List[float]

class PredictionResponse(PydanticBaseModel):
    store_id: int
    date: datetime
    forecast: float

# Load model for inference (if not already loaded via training above)
if model is None:
    try:
        model = joblib.load(config.model_path)
        logger.info("Model loaded for API from %s", config.model_path)
    except Exception as e:
        logger.error("Could not load model for API: %s", e)
        model = None

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predict the sales for a given store and date using the trained model and recent sales data.
    The request should include the store_id, the date for prediction, and a list of recent sales values.
    """
    # Ensure model is available
    if model is None:
        logger.error("Prediction attempted without a loaded model.")
        raise HTTPException(status_code=500, detail="Model not loaded.")
    # Check that sufficient history is provided for features
    needed_history = max(max(config.lag_days) if config.lag_days else 0,
                         max(config.rolling_windows) if config.rolling_windows else 0)
    if len(request.recent_sales) < needed_history:
        msg = f"At least {needed_history} recent sales values are required, got {len(request.recent_sales)}."
        logger.error("Insufficient history for prediction. " + msg)
        raise HTTPException(status_code=400, detail=msg)
    # Construct feature data for prediction
    try:
        # Prepare a dict for features
        feat_data = {
            config.id_col: [request.store_id],
            config.date_col: [request.date]
        }
        # Lag features: assume recent_sales[-1] is yesterday's sales, etc.
        for lag in config.lag_days:
            feat_data[f"lag_{lag}"] = [request.recent_sales[-lag] if lag <= len(request.recent_sales) else None]
        # Rolling mean features
        for window in config.rolling_windows:
            if len(request.recent_sales) >= window:
                feat_data[f"rolling_mean_{window}"] = [pd.Series(request.recent_sales[-window:]).mean()]
            else:
                # If not enough data for full window, use mean of whatever is available
                feat_data[f"rolling_mean_{window}"] = [pd.Series(request.recent_sales).mean()]
        # Date features
        feat_data["day_of_week"] = [request.date.weekday()]
        feat_data["month"] = [request.date.month]
        # Create DataFrame for model input
        input_df = pd.DataFrame(feat_data)
        # Ensure all expected feature columns are present
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_columns]
        # Perform prediction
        prediction = model.predict(input_df)[0]
        logger.info("Prediction made for store_id %s on %s: %.2f", request.store_id, request.date.date(), prediction)
        return PredictionResponse(store_id=request.store_id, date=request.date, forecast=float(prediction))
    except Exception as e:
        logger.error("Error during prediction: %s", e)
        raise HTTPException(status_code=500, detail="Internal prediction error.")
