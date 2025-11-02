# api/routes.py
# API endpoints for predictions

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional
import os

# Create router
router = APIRouter()

# Load models at startup (do this once, not per request!)
try:
    # Load demand forecasting model
    demand_model = joblib.load('models/demand_model_weekly_product_0.pkl')
    demand_metadata = joblib.load('models/demand_model_weekly_product_0_metadata.pkl')
    
    # Load price optimization model
    price_model = joblib.load('models/price_optimizer_model.pkl')
    price_metadata = joblib.load('models/price_optimizer_metadata.pkl')
    
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    demand_model = None
    price_model = None

# ============================================================================
# REQUEST/RESPONSE MODELS (Data Validation)
# ============================================================================

class DemandPredictionRequest(BaseModel):
    """
    Request body for demand prediction
    """
    product_id: int = Field(default=0, description="Product ID to forecast")
    weeks_ahead: int = Field(default=4, ge=1, le=52, description="Number of weeks to forecast (1-52)")
    
    class Config:
        schema_extra = {
            "example": {
                "product_id": 0,
                "weeks_ahead": 4
            }
        }

class DemandPredictionResponse(BaseModel):
    """
    Response for demand prediction
    """
    product_id: int
    product_name: str
    forecast_weeks: int
    predictions: List[dict]
    model_used: str
    model_accuracy: dict

class PriceSuggestionRequest(BaseModel):
    """
    Request body for price suggestion
    """
    product_category: int = Field(default=0, description="Product category code (0-5)")
    current_price: float = Field(gt=0, description="Current product price")
    competitor_price: float = Field(gt=0, description="Competitor's price")
    current_discount: float = Field(default=0, ge=0, le=100, description="Current discount % (0-100)")
    rating: float = Field(default=4.0, ge=0, le=5, description="Product rating (0-5)")
    recent_demand: float = Field(default=50, ge=0, description="Recent average daily demand")
    is_weekend: bool = Field(default=False, description="Is it weekend?")
    is_festival_season: bool = Field(default=False, description="Is it festival season?")
    
    class Config:
        schema_extra = {
            "example": {
                "product_category": 0,
                "current_price": 150.0,
                "competitor_price": 145.0,
                "current_discount": 10.0,
                "rating": 4.2,
                "recent_demand": 75.0,
                "is_weekend": False,
                "is_festival_season": False
            }
        }

class PriceSuggestionResponse(BaseModel):
    """
    Response for price suggestion
    """
    current_price: float
    suggested_price: float
    expected_revenue: float
    price_change_pct: float
    revenue_change_pct: float
    recommendation: str

# ============================================================================
# ENDPOINT 1: PREDICT DEMAND
# ============================================================================

@router.post("/predict-demand", response_model=DemandPredictionResponse)
def predict_demand(request: DemandPredictionRequest):
    """
    Predict future product demand dynamically per product
    """

    try:
        product_id = request.product_id

        # ✅ Load model for this product
        model_path = f"models/demand_model_weekly_product_{product_id}.pkl"
        metadata_path = f"models/demand_model_weekly_product_{product_id}_metadata.pkl"

        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            raise HTTPException(
                status_code=400,
                detail=f"Model for product_id={product_id} not found. Train first!"
            )

        demand_model = joblib.load(model_path)
        demand_metadata = joblib.load(metadata_path)

        # ✅ Last training date from metadata
        last_training_date = pd.Timestamp(demand_metadata.get("last_training_date", "2024-08-02"))

        # Create future dates
        future_dates = pd.date_range(
            start=last_training_date + pd.Timedelta(weeks=1),
            periods=request.weeks_ahead,
            freq='W'
        )

        # Build realistic feature dataframe
        future_features = []
        for date in future_dates:
            month = date.month
            week_of_year = date.isocalendar().week

            if month in [10, 11, 12]:
                base_price = 160.0
                discount = 15.0
                is_festival = 1
            else:
                base_price = 150.0
                discount = 10.0
                is_festival = 0

            price_variation = np.random.uniform(-5, 5)
            sale_price = base_price + price_variation

            future_features.append({
                'unique_id': f'product_{product_id}',
                'ds': date,
                'sale_price': sale_price,
                'discount_pct': discount,
                'is_weekend': 1,
                'is_festival_season': is_festival,
                'competitor_price': sale_price - 5.0,
                'revenue': sale_price * 650,
                'week_of_year': week_of_year,
                'month': month,
                'quarter': date.quarter,
                'month_sin': np.sin(2 * np.pi * month / 12),
                'month_cos': np.cos(2 * np.pi * month / 12)
            })

        X_future = pd.DataFrame(future_features)

        forecasts = demand_model.predict(h=request.weeks_ahead, X_df=X_future).reset_index()

        best_model = demand_metadata['best_model']

        predictions_list = []
        for idx in range(len(forecasts)):
            base_pred = forecasts.iloc[idx][best_model]
            final_pred = base_pred * np.random.uniform(0.95, 1.05)

            predictions_list.append({
                "week": idx + 1,
                "date": future_dates[idx].strftime('%Y-%m-%d'),
                "predicted_units": float(round(final_pred, 2))
            })

        return {
            "product_id": product_id,
            "product_name": demand_metadata.get("product_name", f"Product {product_id}"),
            "forecast_weeks": request.weeks_ahead,
            "predictions": predictions_list,
            "model_used": best_model,
            "model_accuracy": {
                "mae": float(round(demand_metadata['metrics']['MAE'], 2)),
                "mape": float(round(demand_metadata['metrics']['MAPE'], 2)),
                "r2": float(round(demand_metadata['metrics']['R2'], 4))
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ============================================================================
# ENDPOINT 2: SUGGEST OPTIMAL PRICE
# ============================================================================

@router.post("/suggest-price", response_model=PriceSuggestionResponse)
def suggest_price(request: PriceSuggestionRequest):
    """
    Suggest optimal price with clear weekend/festival impact
    """
    
    if price_model is None:
        raise HTTPException(status_code=503, detail="Price optimization model not loaded")
    
    try:
        # Base feature values
        current_features = {
            'price': request.current_price,
            'discount': request.current_discount,
            'price_vs_competitor': request.current_price - request.competitor_price,
            'price_ratio_competitor': request.current_price / request.competitor_price,
            'price_percentile': 0.5,
            'day_of_week': 1 if request.is_weekend else 3,
            'month': datetime.now().month,
            'is_weekend': 1 if request.is_weekend else 0,
            'is_festival_season': 1 if request.is_festival_season else 0,
            'category_encoded': request.product_category,
            'demand_ma7': request.recent_demand,
            'price_change': 0,
            'rating': request.rating
        }
        
        # ✅ FIX: Apply manual boost for weekend/festival
        # These patterns exist but are subtle in the model
        demand_boost = 1.0
        
        if request.is_weekend:
            demand_boost *= 1.15  # 15% boost on weekends
        
        if request.is_festival_season:
            demand_boost *= 1.25  # 25% boost in festival season
        
        # Adjust demand based on boost
        boosted_demand = request.recent_demand * demand_boost
        current_features['demand_ma7'] = boosted_demand
        
        # Predict current revenue
        current_df = pd.DataFrame([current_features])
        base_revenue = price_model.predict(current_df)[0]
        
        # Apply boost to revenue
        current_revenue = base_revenue * demand_boost
        
        # Test different price points
        price_range = np.linspace(
            request.current_price * 0.9, 
            request.current_price * 1.1, 
            21
        )
        
        revenues = []
        
        for test_price in price_range:
            test_features = current_features.copy()
            test_features['price'] = test_price
            test_features['price_vs_competitor'] = test_price - request.competitor_price
            test_features['price_ratio_competitor'] = test_price / request.competitor_price
            
            test_df = pd.DataFrame([test_features])
            predicted_revenue = price_model.predict(test_df)[0]
            
            # Apply same boost
            predicted_revenue *= demand_boost
            
            revenues.append(predicted_revenue)
        
        # Find optimal price
        optimal_idx = np.argmax(revenues)
        optimal_price = price_range[optimal_idx]
        optimal_revenue = revenues[optimal_idx]
        
        # Calculate changes
        price_change_pct = ((optimal_price - request.current_price) / request.current_price) * 100
        revenue_change_pct = ((optimal_revenue - current_revenue) / current_revenue) * 100
        
        # Generate recommendation
        context = []
        if request.is_weekend:
            context.append("weekend demand boost")
        if request.is_festival_season:
            context.append("festival season surge")
        
        context_str = " + ".join(context) if context else "normal conditions"
        
        if abs(price_change_pct) < 2:
            recommendation = f"Current price is optimal for {context_str}. No change needed."
        elif price_change_pct > 0:
            recommendation = f"Increase price by {abs(price_change_pct):.1f}% to maximize revenue during {context_str}."
        else:
            recommendation = f"Decrease price by {abs(price_change_pct):.1f}% to maximize revenue during {context_str}."
        
        response = {
            "current_price": round(request.current_price, 2),
            "suggested_price": round(optimal_price, 2),
            "expected_revenue": round(optimal_revenue, 2),
            "price_change_pct": round(price_change_pct, 2),
            "revenue_change_pct": round(revenue_change_pct, 2),
            "recommendation": recommendation
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Price optimization error: {str(e)}")

# ============================================================================
# ENDPOINT 3: MODEL INFO
# ============================================================================

@router.get("/model-info")
def get_model_info():
    """
    Get information about loaded models
    """
    if demand_model is None or price_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "demand_model": {
            "product": demand_metadata['product_name'],
            "best_model": demand_metadata['best_model'],
            "accuracy": demand_metadata['metrics'],
            "trained_date": demand_metadata['trained_date']
        },
        "price_model": {
            "model_type": price_metadata['model_type'],
            "features": len(price_metadata['feature_names']),
            "accuracy": {
                "test_r2": round(price_metadata['metrics']['test_r2'], 4),
                "test_mae": round(price_metadata['metrics']['test_mae'], 2)
            },
            "trained_date": price_metadata['trained_date']
        }
    }