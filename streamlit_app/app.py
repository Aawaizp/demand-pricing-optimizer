# streamlit_app/app.py
# Beautiful Streamlit dashboard for demand forecasting and price optimization

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Demand & Price Optimizer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_URL = "http://127.0.0.1:8000"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_api_health():
    """Check if FastAPI backend is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def call_predict_demand(product_id, weeks_ahead):
    """Call demand prediction API"""
    try:
        payload = {
            "product_id": product_id,
            "weeks_ahead": weeks_ahead
        }
        response = requests.post(f"{API_URL}/predict-demand", json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def call_suggest_price(params):
    """Call price optimization API"""
    try:
        response = requests.post(f"{API_URL}/suggest-price", json=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def get_model_info():
    """Get model information"""
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Title and description
    st.title("🤖 AI-Powered Demand Forecasting & Price Optimization")
    st.markdown("### Predict future demand and optimize pricing for maximum revenue using Machine Learning")
    
    # Check API health
    api_status = check_api_health()
    
    if not api_status:
        st.error("⚠️ **FastAPI Backend Not Running!**")
        st.info("Please start the API server first:")
        st.code("uvicorn api.main:app --reload", language="bash")
        st.stop()
    else:
        st.success("✅ API Connected")
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Choose a feature:",
        ["🏠 Home", "📈 Demand Forecasting", "💰 Price Optimization", "📊 Model Performance"]
    )
    
    # ========================================================================
    # PAGE 1: HOME
    # ========================================================================
    
    if page == "🏠 Home":
        st.markdown("---")
        
        # Hero section with better styling
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📈 Demand Forecasting")
            st.info("""
            Predict future product demand using StatsForecast models (ARIMA, ETS, Theta)
            
            - Weekly forecasts
            - Multiple time horizons
            - Seasonal patterns
            """)
        
        with col2:
            st.markdown("### 💰 Price Optimization")
            st.success("""
            Find optimal pricing using XGBoost regression with SHAP explainability
            
            - Revenue maximization
            - Competitor analysis
            - Discount optimization
            """)
        
        with col3:
            st.markdown("### 📊 Model Performance")
            st.warning("""
            Track model accuracy and feature importance
            
            - Real-time metrics
            - SHAP visualizations
            - Model explainability
            """)
        
        st.markdown("---")
        
        # Quick stats
        st.subheader("🎯 Quick Stats")
        
        model_info = get_model_info()
        
        if model_info:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Demand Model",
                    model_info['demand_model']['best_model'],
                    f"MAPE: {model_info['demand_model']['accuracy']['MAPE']:.2f}%"
                )
            
            with col2:
                st.metric(
                    "Demand Accuracy (R²)",
                    f"{model_info['demand_model']['accuracy']['R2']:.2%}",
                    "Test Set"
                )
            
            with col3:
                st.metric(
                    "Price Model",
                    "XGBoost",
                    f"R²: {model_info['price_model']['accuracy']['test_r2']:.2%}"
                )
            
            with col4:
                st.metric(
                    "Features Used",
                    model_info['price_model']['features'],
                    "Pricing Model"
                )
        
        st.markdown("---")
        
        # How to use
        st.subheader("📖 How to Use")
        
        st.markdown("""
        1. **Demand Forecasting**: Navigate to the Demand Forecasting page to predict future sales
        2. **Price Optimization**: Go to Price Optimization to find the best price point
        3. **Model Performance**: Check model accuracy and insights
        
        **Tech Stack**: Python, StatsForecast, XGBoost, FastAPI, Streamlit, SHAP
        """)
    
    # ========================================================================
    # PAGE 2: DEMAND FORECASTING
    # ========================================================================
    
    elif page == "📈 Demand Forecasting":
        st.header("📈 Demand Forecasting")
        st.markdown("Predict future product demand for strategic planning")
        
        st.markdown("---")
        
        # Input section
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Forecast Settings")
            
           
            
            product_id = st.number_input(
                "Product ID",
                min_value=0,
                max_value=19,  # Only 0 is available
                value=0,
                
                help="Select product to forecast (0-19)"
            )
            
            weeks_ahead = st.slider(
                "Forecast Horizon (weeks)",
                min_value=1,
                max_value=12,
                value=4,
                help="How many weeks to forecast"
            )
            
            forecast_button = st.button("🔮 Generate Forecast", type="primary")
        
        with col2:
            st.subheader("📊 Forecast Results")
            
            if forecast_button:
                with st.spinner("Generating forecast..."):
                    result = call_predict_demand(product_id, weeks_ahead)
                
                if result:
                    st.success(f"✅ Forecast generated for: **{result['product_name']}**")
                    
                    # Display metrics
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric("Model Used", result['model_used'])
                    
                    with col_b:
                        st.metric("MAPE", f"{result['model_accuracy']['mape']:.2f}%")
                    
                    with col_c:
                        st.metric("R² Score", f"{result['model_accuracy']['r2']:.4f}")
                    
                    # Create DataFrame from predictions
                    pred_df = pd.DataFrame(result['predictions'])
                    
                    # Display table
                    st.subheader("📅 Weekly Predictions")
                    st.dataframe(pred_df, use_container_width=True)
                    
                    # Plot forecast
                    st.subheader("📈 Forecast Visualization")
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=pred_df['date'],
                        y=pred_df['predicted_units'],
                        mode='lines+markers',
                        name='Predicted Demand',
                        line=dict(color='#4CAF50', width=3),
                        marker=dict(size=10)
                    ))
                    
                    fig.update_layout(
                        title=f"Weekly Demand Forecast - {result['product_name']}",
                        xaxis_title="Week",
                        yaxis_title="Predicted Units",
                        hovermode='x unified',
                        height=500,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download option
                    csv = pred_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Forecast (CSV)",
                        data=csv,
                        file_name=f"demand_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
    
    # ========================================================================
    # PAGE 3: PRICE OPTIMIZATION
    # ========================================================================
    
    elif page == "💰 Price Optimization":
        st.header("💰 Price Optimization")
        st.markdown("Find the optimal price to maximize revenue")
        
        st.markdown("---")
        
        # Input section
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Product Details")
            
            category_map = {
                "Beauty & Hygiene": 0,
                "Beverages": 1,
                "Cleaning & Household": 2,
                "Foodgrains, Oil & Masala": 3,
                "Snacks & Branded Foods": 4,
                "Kitchen, Garden & Pets": 5
            }
            
            category_name = st.selectbox(
                "Product Category",
                list(category_map.keys())
            )
            
            current_price = st.number_input(
                "Current Price (₹)",
                min_value=10.0,
                max_value=1000.0,
                value=150.0,
                step=5.0
            )
            
            competitor_price = st.number_input(
                "Competitor Price (₹)",
                min_value=10.0,
                max_value=1000.0,
                value=145.0,
                step=5.0
            )
            
            current_discount = st.slider(
                "Current Discount (%)",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=1.0
            )
            
            rating = st.slider(
                "Product Rating",
                min_value=0.0,
                max_value=5.0,
                value=4.0,
                step=0.1
            )
            
            recent_demand = st.number_input(
                "Recent Daily Demand",
                min_value=1.0,
                max_value=500.0,
                value=75.0,
                step=5.0
            )
            
            is_weekend = st.checkbox("Is Weekend?", value=False)
            is_festival = st.checkbox("Is Festival Season?", value=False)
            
            optimize_button = st.button("🎯 Optimize Price", type="primary")
        
        with col2:
            st.subheader("📊 Optimization Results")
            
            if optimize_button:
                with st.spinner("Optimizing price..."):
                    params = {
                        "product_category": category_map[category_name],
                        "current_price": current_price,
                        "competitor_price": competitor_price,
                        "current_discount": current_discount,
                        "rating": rating,
                        "recent_demand": recent_demand,
                        "is_weekend": is_weekend,
                        "is_festival_season": is_festival
                    }
                    
                    result = call_suggest_price(params)
                
                if result:
                    # Display recommendation
                    if result['price_change_pct'] > 0:
                        st.success(f"💡 **Recommendation:** {result['recommendation']}")
                    elif result['price_change_pct'] < 0:
                        st.warning(f"💡 **Recommendation:** {result['recommendation']}")
                    else:
                        st.info(f"💡 **Recommendation:** {result['recommendation']}")
                    
                    # Metrics
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric(
                            "Current Price",
                            f"₹{result['current_price']:.2f}"
                        )
                    
                    with col_b:
                        st.metric(
                            "Suggested Price",
                            f"₹{result['suggested_price']:.2f}",
                            f"{result['price_change_pct']:+.1f}%"
                        )
                    
                    with col_c:
                        st.metric(
                            "Expected Revenue",
                            f"₹{result['expected_revenue']:.2f}",
                            f"{result['revenue_change_pct']:+.1f}%"
                        )
                    
                    # Visualization with ACTUAL revenue curve
                    st.subheader("📈 Price Impact Analysis")
                    
                    # Simulate revenue curve by calling API for multiple prices
                    price_range = np.linspace(current_price * 0.8, current_price * 1.2, 21)
                    revenues = []
                    
                    with st.spinner("Calculating revenue curve..."):
                        for test_price in price_range:
                            test_params = params.copy()
                            test_params['current_price'] = float(test_price)
                            test_result = call_suggest_price(test_params)
                            if test_result:
                                revenues.append(test_result['expected_revenue'])
                            else:
                                revenues.append(0)
                    
                    # Plot the curve
                    fig = go.Figure()
                    
                    # Revenue curve
                    fig.add_trace(go.Scatter(
                        x=price_range,
                        y=revenues,
                        mode='lines',
                        name='Expected Revenue',
                        line=dict(color='blue', width=3),
                        fill='tozeroy',
                        fillcolor='rgba(0,100,255,0.1)'
                    ))
                    
                    # Optimal price marker
                    fig.add_trace(go.Scatter(
                        x=[result['suggested_price']],
                        y=[result['expected_revenue']],
                        mode='markers',
                        name='Optimal Price',
                        marker=dict(color='green', size=15, symbol='star')
                    ))
                    
                    # Current price marker
                    current_revenue = revenues[10]  # Middle point
                    fig.add_trace(go.Scatter(
                        x=[result['current_price']],
                        y=[current_revenue],
                        mode='markers',
                        name='Current Price',
                        marker=dict(color='red', size=12, symbol='circle')
                    ))
                    
                    fig.update_layout(
                        title="Price vs Revenue Curve",
                        xaxis_title="Price (₹)",
                        yaxis_title="Expected Revenue (₹)",
                        height=500,
                        template='plotly_white',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # PAGE 4: MODEL PERFORMANCE
    # ========================================================================
    
    elif page == "📊 Model Performance":
        st.header("📊 Model Performance & Insights")
        
        model_info = get_model_info()
        
        if model_info:
            # Demand Model
            st.subheader("📈 Demand Forecasting Model")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Model Details:**")
                st.write(f"- Product: {model_info['demand_model']['product']}")
                st.write(f"- Best Model: {model_info['demand_model']['best_model']}")
                st.write(f"- Trained: {model_info['demand_model']['trained_date']}")
            
            with col2:
                st.markdown("**Accuracy Metrics:**")
                st.write(f"- MAE: {model_info['demand_model']['accuracy']['MAE']:.2f} units")
                st.write(f"- MAPE: {model_info['demand_model']['accuracy']['MAPE']:.2f}%")
                st.write(f"- R²: {model_info['demand_model']['accuracy']['R2']:.4f}")
            
            st.markdown("---")
            
            # Price Model
            st.subheader("💰 Price Optimization Model")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Model Details:**")
                st.write(f"- Type: {model_info['price_model']['model_type']}")
                st.write(f"- Features: {model_info['price_model']['features']}")
                st.write(f"- Trained: {model_info['price_model']['trained_date']}")
            
            with col2:
                st.markdown("**Accuracy Metrics:**")
                st.write(f"- Test R²: {model_info['price_model']['accuracy']['test_r2']:.4f}")
                st.write(f"- Test MAE: ₹{model_info['price_model']['accuracy']['test_mae']:.2f}")
            
            st.markdown("---")
            
            # SHAP visualizations
            st.subheader("🔍 Model Explainability (SHAP)")
            
            st.markdown("SHAP (SHapley Additive exPlanations) shows which features impact predictions the most:")
            
            # Display SHAP images
            try:
                st.image("models/shap_summary_plot.png", caption="SHAP Summary Plot - Feature Importance")
                st.image("models/feature_importance.png", caption="XGBoost Feature Importance")
            except:
                st.info("SHAP plots not found. Run `python src/shap_explainer.py` to generate them.")
        else:
            st.error("Could not load model information")

# Run the app
if __name__ == "__main__":
    main()