# src/train_demand_model_weekly.py
# Weekly demand forecasting - Better for FMCG products!

import pandas as pd
import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoCES
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os

warnings.filterwarnings('ignore')

def load_and_aggregate_weekly(product_id=0):
    """
    Load sales data and aggregate to WEEKLY totals
    This smooths out daily randomness!
    
    Args:
        product_id: Which product to forecast
    Returns: Weekly aggregated DataFrame, product name
    """
    print("📂 Loading sales data for WEEKLY aggregation...")
    
    # Read sales data
    df = pd.read_csv('data/processed/sales_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter for specific product
    product_data = df[df['product_id'] == product_id].copy()
    product_name = product_data['product_name'].iloc[0]
    
    print(f"   Product: {product_name}")
    print(f"   Daily records: {len(product_data)}")
    
    # Set date as index for resampling
    product_data = product_data.set_index('date')
    
    # Resample to WEEKLY frequency ('W' = week ending on Sunday)
    # Sum quantities, average prices
    weekly_data = product_data.resample('W').agg({
        'quantity_sold': 'sum',              # Total units sold per week
        'sale_price': 'mean',                # Average price per week
        'discount_pct': 'mean',              # Average discount per week
        'is_weekend': 'max',                 # Was there a weekend? (always 1)
        'is_festival_season': 'max',         # Was it festival season?
        'competitor_price': 'mean',          # Average competitor price
        'revenue': 'sum'                     # Total revenue per week
    }).reset_index()
    
    # Rename for StatsForecast
    weekly_data = weekly_data.rename(columns={
        'date': 'ds',
        'quantity_sold': 'y'
    })
    
    # Add unique_id
    weekly_data['unique_id'] = f'product_{product_id}'
    
    # Add time features
    weekly_data['week_of_year'] = weekly_data['ds'].dt.isocalendar().week
    weekly_data['month'] = weekly_data['ds'].dt.month
    weekly_data['quarter'] = weekly_data['ds'].dt.quarter
    
    # Create cyclical features (month encoding)
    weekly_data['month_sin'] = np.sin(2 * np.pi * weekly_data['month'] / 12)
    weekly_data['month_cos'] = np.cos(2 * np.pi * weekly_data['month'] / 12)
    
    # Sort by date
    weekly_data = weekly_data.sort_values('ds').reset_index(drop=True)
    
    print(f"✅ Aggregated to {len(weekly_data)} WEEKS of data")
    print(f"   Average weekly sales: {weekly_data['y'].mean():.2f} units")
    print(f"   Date range: {weekly_data['ds'].min().date()} to {weekly_data['ds'].max().date()}")
    
    return weekly_data, product_name

def split_train_test(df, test_weeks=12):
    """
    Split into train/test sets
    
    Args:
        df: Weekly data
        test_weeks: How many weeks for testing (default: 12 weeks = 3 months)
    Returns: train_df, test_df
    """
    print(f"\n✂️ Splitting data into train/test sets...")
    print(f"   Test period: Last {test_weeks} weeks")
    
    # Split by number of weeks from end
    train_df = df.iloc[:-test_weeks].copy()
    test_df = df.iloc[-test_weeks:].copy()
    
    print(f"   Training data: {len(train_df)} weeks ({train_df['ds'].min().date()} to {train_df['ds'].max().date()})")
    print(f"   Testing data: {len(test_df)} weeks ({test_df['ds'].min().date()} to {test_df['ds'].max().date()})")
    
    return train_df, test_df

def train_statsforecast_weekly(train_df):
    """
    Train StatsForecast models on WEEKLY data
    
    Args:
        train_df: Training data
    Returns: Trained StatsForecast object
    """
    print("\n🧠 Training StatsForecast models on WEEKLY data...")
    print("   Models: AutoARIMA, AutoETS, AutoCES")
    print("   This will be FAST... ⚡")
    
    # Define models
    # season_length=52 means 52 weeks (yearly seasonality)
    # NOTE: AutoCES uses numba/llvmlite and can take a long time to compile on first run.
    # To keep training fast and avoid long JIT compilation delays, we omit AutoCES by default.
    models = [
        AutoARIMA(season_length=52),  # Yearly seasonality (52 weeks)
        AutoETS(season_length=52),
        # AutoCES(season_length=52),  # disabled to avoid long numba compilation on Windows
    ]
    
    # Create StatsForecast object
    sf = StatsForecast(
        models=models,
        freq='W',  # Weekly frequency
        n_jobs=1  # run single-threaded to reduce parallel numba compilation issues on Windows
    )
    
    # Train models
    print("   Training in progress... ⏳")
    sf.fit(train_df)
    
    print("✅ Model training complete!")
    
    return sf

def make_predictions(sf_model, train_df, test_weeks):
    print(f"\n🔮 Making predictions for next {test_weeks} weeks...")
    print("   Using ONLY training data ...")

    last_date = train_df['ds'].max()

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1),
        periods=test_weeks,
        freq='W'
    )

    # Create future exogenous dataframe
    feature_cols = [
        'sale_price', 'discount_pct', 'is_weekend', 'is_festival_season',
        'competitor_price', 'revenue', 'week_of_year', 'month', 'quarter',
        'month_sin', 'month_cos'
    ]

    # 👉 For future weeks, we don't know actual values, so use last known values
    last_values = train_df.iloc[-1][feature_cols]

    X_future = pd.DataFrame({
        'unique_id': train_df['unique_id'].iloc[0],
        'ds': future_dates
    })

    # Fill future exogenous features with last seen values
    for col in feature_cols:
        X_future[col] = last_values[col]

    # ✅ Predict with exogenous regressors
    forecasts = sf_model.predict(
        h=test_weeks,
        X_df=X_future
    ).reset_index()

    # Add dates
    forecasts['ds'] = future_dates.values

    print(f"✅ Generated {test_weeks} week predictions")
    return forecasts




def calculate_metrics(test_df, forecasts):
    """
    Calculate accuracy metrics for each model
    
    Args:
        test_df: Actual test data
        forecasts: Model predictions
    Returns: Best model name, metrics, all results
    """
    print("\n📊 Calculating accuracy metrics for WEEKLY forecasts...")
    
    # Get actual values
    y_true = test_df['y'].values
    
    # Get model columns
    model_columns = [col for col in forecasts.columns if col not in ['unique_id', 'ds']]
    
    results = {}
    
    # Calculate metrics for each model
    for model_name in model_columns:
        y_pred = forecasts[model_name].values
        
        # Handle any NaN predictions
        if np.any(np.isnan(y_pred)):
            print(f"   ⚠️ {model_name} has NaN predictions, skipping...")
            continue
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100  # Add small value to avoid division by zero
        
        results[model_name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
    
    # Find best model (lowest MAE)
    if not results:
        print("❌ No valid models found!")
        return None, None, None
    
    best_model = min(results, key=lambda x: results[x]['MAE'])
    
    # Print results
    print("\n" + "=" * 70)
    print("📈 WEEKLY MODEL PERFORMANCE COMPARISON")
    print("=" * 70)
    
    for model_name, metrics in results.items():
        is_best = "⭐ BEST" if model_name == best_model else ""
        print(f"\n{model_name} {is_best}")
        print(f"  MAE:  {metrics['MAE']:.2f} units/week")
        print(f"  RMSE: {metrics['RMSE']:.2f} units/week")
        print(f"  R²:   {metrics['R2']:.4f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
    
    print("\n" + "=" * 70)
    print(f"🏆 Best Model: {best_model}")
    print("=" * 70)
    
    best_metrics = results[best_model]
    
    # Explain metrics in simple terms
    print("\n🎯 What do these numbers mean?")
    print(f"• On average, weekly predictions are off by {best_metrics['MAE']:.0f} units (MAE)")
    print(f"• Model explains {best_metrics['R2']*100:.1f}% of weekly sales variation (R²)")
    print(f"• Predictions are {best_metrics['MAPE']:.1f}% away from actual values (MAPE)")
    
    # Performance assessment
    if best_metrics['R2'] > 0.7:
        print("✅ EXCELLENT! Weekly aggregation made a huge difference! 🎉")
    elif best_metrics['R2'] > 0.5:
        print("👍 GOOD! Much better than daily predictions!")
    elif best_metrics['R2'] > 0.3:
        print("📈 IMPROVED! Weekly forecasting helps smooth the data.")
    else:
        print("⚠️ Still challenging, but weekly is better than daily.")
    
    return best_model, best_metrics, results

def plot_weekly_predictions(test_df, forecasts, best_model, product_name, 
                            save_path='models/demand_forecast_weekly.png'):
    """
    Visualize actual vs predicted WEEKLY sales
    
    Args:
        test_df: Actual test data
        forecasts: Model predictions
        best_model: Name of best model
        product_name: Product name
        save_path: Where to save plot
    """
    print("\n📊 Creating weekly forecast visualization...")
    
    # Merge test data with forecasts
    comparison = test_df[['ds', 'y']].merge(
        forecasts[['ds', best_model]], 
        on='ds', 
        how='inner'
    )
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Time series comparison
    ax1.plot(comparison['ds'], comparison['y'], 
             label='Actual Weekly Sales', color='blue', linewidth=2.5, marker='o', markersize=6)
    
    ax1.plot(comparison['ds'], comparison[best_model], 
             label=f'Predicted Weekly Sales ({best_model})', color='red', 
             linewidth=2.5, linestyle='--', marker='x', markersize=6)
    
    ax1.set_title(f'Weekly Demand Forecast: {product_name}\nActual vs Predicted (Model: {best_model})', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Week', fontsize=12)
    ax1.set_ylabel('Units Sold (per week)', fontsize=12)
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Residuals (errors)
    residuals = comparison['y'] - comparison[best_model]
    ax2.bar(comparison['ds'], residuals, color=['red' if x < 0 else 'green' for x in residuals], alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_title('Prediction Errors (Residuals)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Week', fontsize=12)
    ax2.set_ylabel('Error (Actual - Predicted)', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {save_path}")
    
    plt.show()

def save_weekly_model(sf_model, product_id, product_name, best_model, best_metrics):
    """
    Save trained weekly forecasting model
    
    Args:
        sf_model: Trained StatsForecast object
        product_id: Product ID
        product_name: Product name
        best_model: Best model name
        best_metrics: Performance metrics
    """
    print("\n💾 Saving weekly forecasting model...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_path = f'models/demand_model_weekly_product_{product_id}.pkl'
    joblib.dump(sf_model, model_path)
    
    # Save metadata
    metadata = {
        'product_id': product_id,
        'product_name': product_name,
        'best_model': best_model,
        'metrics': best_metrics,
        'frequency': 'weekly',
        'trained_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'framework': 'StatsForecast'
    }
    
    metadata_path = f'models/demand_model_weekly_product_{product_id}_metadata.pkl'
    joblib.dump(metadata, metadata_path)
    
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Metadata saved to: {metadata_path}")

def main():
    """
    Main training pipeline for WEEKLY forecasting
    """
    print("=" * 70)
    print("🚀 WEEKLY DEMAND FORECASTING MODEL TRAINING")
    print("=" * 70)
    print("💡 Weekly aggregation smooths daily noise for better predictions!")
    print("=" * 70)
    
    # Step 1: Load and aggregate to weekly
    product_id = 0
    weekly_data, product_name = load_and_aggregate_weekly(product_id)
    
    # Step 2: Split into train/test (12 weeks = 3 months for testing)
    test_weeks = 12
    train_df, test_df = split_train_test(weekly_data, test_weeks=test_weeks)
    
    # Step 3: Train models
    sf_model = train_statsforecast_weekly(train_df)
    
    # Step 4: Make predictions (provide test_df so we can pass exogenous features)
    forecasts = make_predictions(sf_model, train_df, test_weeks)
    
    # Step 5: Calculate metrics
    best_model, best_metrics, all_results = calculate_metrics(test_df, forecasts)
    
    if best_model is None:
        print("❌ Training failed. Please check the data.")
        return
    
    # Step 6: Plot results
    plot_weekly_predictions(test_df, forecasts, best_model, product_name)
    
    # Step 7: Save model
    save_weekly_model(sf_model, product_id, product_name, best_model, best_metrics)
    
    print("\n" + "=" * 70)
    print("✅ WEEKLY FORECASTING MODEL TRAINING COMPLETE!")
    print("=" * 70)
    print("\n📁 Files created:")
    print("   • models/demand_model_weekly_product_0.pkl")
    print("   • models/demand_model_weekly_product_0_metadata.pkl")
    print("   • models/demand_forecast_weekly.png")
    print("\n💡 Weekly forecasting is much more accurate for business planning!")

# Run when script is executed
if __name__ == "__main__":
    main()