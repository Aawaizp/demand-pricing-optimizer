# src/train_price_model.py
# Train XGBoost model to optimize product pricing for maximum revenue

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

def load_sales_data():
    """
    Load the processed sales data
    Returns: DataFrame with all sales data
    """
    print("📂 Loading sales data for price optimization...")
    
    # Read sales data
    df = pd.read_csv('data/processed/sales_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"✅ Loaded {len(df)} sales records")
    print(f"   Products: {df['product_id'].nunique()}")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    return df

def prepare_pricing_features(df):
    """
    Create features for price optimization model
    
    The goal: Predict revenue based on price and other factors
    Then we can find the optimal price!
    
    Args:
        df: Sales DataFrame
    Returns: DataFrame with features for modeling
    """
    print("\n🔧 Engineering features for price optimization...")
    
    # Create a copy
    pricing_df = df.copy()
    
    # Target variable: Revenue per transaction
    # We want to maximize this!
    pricing_df['revenue_per_unit'] = pricing_df['revenue'] / pricing_df['quantity_sold']
    
    # Feature 1: Price point
    pricing_df['price'] = pricing_df['sale_price']
    
    # Feature 2: Discount percentage
    pricing_df['discount'] = pricing_df['discount_pct']
    
    # Feature 3: Competitor price difference
    pricing_df['price_vs_competitor'] = pricing_df['sale_price'] - pricing_df['competitor_price']
    pricing_df['price_ratio_competitor'] = pricing_df['sale_price'] / (pricing_df['competitor_price'] + 0.01)
    
    # Feature 4: Price elasticity indicator (is it expensive or cheap?)
    # Calculate price percentile within each product
    pricing_df['price_percentile'] = pricing_df.groupby('product_id')['sale_price'].rank(pct=True)
    
    # Feature 5: Time-based features
    pricing_df['day_of_week'] = pricing_df['day_of_week']
    pricing_df['month'] = pricing_df['month']
    pricing_df['is_weekend'] = pricing_df['is_weekend']
    pricing_df['is_festival_season'] = pricing_df['is_festival_season']
    
    # Feature 6: Product category (encode as numeric)
    pricing_df['category_encoded'] = pricing_df['category'].astype('category').cat.codes
    
    # Feature 7: Historical demand (quantity sold)
    pricing_df['demand'] = pricing_df['quantity_sold']
    
    # Feature 8: Moving average of past sales (smooth out noise)
    pricing_df['demand_ma7'] = pricing_df.groupby('product_id')['quantity_sold'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    
    # Feature 9: Price change from previous day
    pricing_df['price_change'] = pricing_df.groupby('product_id')['sale_price'].diff().fillna(0)
    
    # Feature 10: Product rating
    pricing_df['rating'] = pricing_df['rating']
    
    print(f"✅ Created {pricing_df.shape[1]} features")
    
    return pricing_df

def select_features_and_target(df):
    """
    Select final feature set and target variable
    
    Args:
        df: DataFrame with all features
    Returns: X (features), y (target), feature_names
    """
    print("\n🎯 Selecting features and target variable...")
    
    # Features to use for prediction
    feature_columns = [
        'price',                    # Current price (most important!)
        'discount',                 # Discount offered
        'price_vs_competitor',      # How we compare to competitors
        'price_ratio_competitor',   # Price ratio vs competitor
        'price_percentile',         # Is this price high or low for this product?
        'day_of_week',              # Day of week effect
        'month',                    # Seasonal effect
        'is_weekend',               # Weekend boost
        'is_festival_season',       # Festival season boost
        'category_encoded',         # Product category
        'demand_ma7',               # Recent demand trend
        'price_change',             # Price change impact
        'rating'                    # Product quality indicator
    ]
    
    # Target: Revenue (what we want to maximize!)
    target = 'revenue'
    
    # Remove any rows with missing values
    df_clean = df[feature_columns + [target]].dropna()
    
    # Separate features and target
    X = df_clean[feature_columns]
    y = df_clean[target]
    
    print(f"✅ Selected {len(feature_columns)} features")
    print(f"   Target: {target}")
    print(f"   Clean samples: {len(df_clean)}")
    
    return X, y, feature_columns

def split_data(X, y, test_size=0.2):
    """
    Split data into training and testing sets
    
    Args:
        X: Features
        y: Target
        test_size: Proportion for testing (default 20%)
    Returns: X_train, X_test, y_train, y_test
    """
    print(f"\n✂️ Splitting data (test size: {test_size*100}%)...")
    
    # Split data randomly (shuffle=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=42,
        shuffle=True
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def train_xgboost_model(X_train, y_train):
    """
    Train XGBoost regression model
    
    XGBoost is perfect for pricing because:
    - Handles non-linear relationships (price elasticity curves)
    - Fast training and prediction
    - Great with tabular data
    - Built-in feature importance
    
    Args:
        X_train: Training features
        y_train: Training target
    Returns: Trained XGBoost model
    """
    print("\n🧠 Training XGBoost model for price optimization...")
    print("   This may take 1-2 minutes... ⏳")
    
    # Create XGBoost regressor
    # Parameters explained:
    # - n_estimators: Number of trees (more = better but slower)
    # - max_depth: Tree depth (controls complexity)
    # - learning_rate: How fast model learns (lower = more accurate but slower)
    # - subsample: Fraction of data to use per tree (prevents overfitting)
    # - colsample_bytree: Fraction of features to use per tree
    
    model = xgb.XGBRegressor(
        n_estimators=200,           # 200 decision trees
        max_depth=6,                # Max depth of each tree
        learning_rate=0.1,          # Learning rate
        subsample=0.8,              # Use 80% of data per tree
        colsample_bytree=0.8,       # Use 80% of features per tree
        random_state=42,
        n_jobs=-1                   # Use all CPU cores
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    print("✅ Model training complete!")
    
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_train, X_test: Feature sets
        y_train, y_test: Target values
    Returns: Dictionary with metrics
    """
    print("\n📊 Evaluating model performance...")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics for training set
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Calculate metrics for test set
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Print results
    print("\n" + "=" * 70)
    print("📈 MODEL PERFORMANCE METRICS")
    print("=" * 70)
    print("\n🏋️ TRAINING SET:")
    print(f"  MAE:  ₹{train_mae:.2f}")
    print(f"  RMSE: ₹{train_rmse:.2f}")
    print(f"  R²:   {train_r2:.4f}")
    
    print("\n🎯 TEST SET (UNSEEN DATA):")
    print(f"  MAE:  ₹{test_mae:.2f}")
    print(f"  RMSE: ₹{test_rmse:.2f}")
    print(f"  R²:   {test_r2:.4f}")
    print("=" * 70)
    
    # Check for overfitting
    r2_diff = train_r2 - test_r2
    if r2_diff < 0.05:
        print("\n✅ Great! Model generalizes well (no overfitting)")
    elif r2_diff < 0.15:
        print("\n👍 Good! Slight overfitting but acceptable")
    else:
        print("\n⚠️ Warning: Model may be overfitting (train R² much higher than test R²)")
    
    print(f"\n💡 Model explains {test_r2*100:.1f}% of revenue variation")
    
    metrics = {
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2
    }
    
    return metrics, y_test_pred

def plot_feature_importance(model, feature_names, save_path='models/feature_importance.png'):
    """
    Plot which features are most important for pricing decisions
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        save_path: Where to save plot
    """
    print("\n📊 Creating feature importance plot...")
    
    # Get feature importance scores
    importance = model.feature_importances_
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Horizontal bar chart
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Feature Importance for Price Optimization\n(Which factors matter most for revenue?)', 
              fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Feature importance plot saved to: {save_path}")
    
    # Print top 5 features
    print("\n🏆 Top 5 Most Important Features:")
    for idx, row in importance_df.head(5).iterrows():
        print(f"   {row['Feature']}: {row['Importance']:.4f}")
    
    plt.show()

def plot_actual_vs_predicted(y_test, y_pred, save_path='models/actual_vs_predicted.png'):
    """
    Plot actual vs predicted revenue
    
    Args:
        y_test: Actual revenue values
        y_pred: Predicted revenue values
        save_path: Where to save plot
    """
    print("\n📊 Creating actual vs predicted plot...")
    
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(y_test, y_pred, alpha=0.5, s=20)
    
    # Perfect prediction line (45-degree line)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Revenue (₹)', fontsize=12)
    plt.ylabel('Predicted Revenue (₹)', fontsize=12)
    plt.title('Actual vs Predicted Revenue\n(Closer to red line = better predictions)', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Actual vs predicted plot saved to: {save_path}")
    
    plt.show()

def save_model(model, feature_names, metrics):
    """
    Save trained model and metadata
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        metrics: Performance metrics
    """
    print("\n💾 Saving price optimization model...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_path = 'models/price_optimizer_model.pkl'
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata = {
        'feature_names': feature_names,
        'metrics': metrics,
        'model_type': 'XGBoost Regressor',
        'target': 'revenue',
        'trained_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metadata_path = 'models/price_optimizer_metadata.pkl'
    joblib.dump(metadata, metadata_path)
    
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Metadata saved to: {metadata_path}")

def main():
    """
    Main training pipeline for price optimization
    """
    print("=" * 70)
    print("🚀 XGBOOST PRICE OPTIMIZATION MODEL TRAINING")
    print("=" * 70)
    
    # Step 1: Load data
    sales_df = load_sales_data()
    
    # Step 2: Engineer features
    pricing_df = prepare_pricing_features(sales_df)
    
    # Step 3: Select features and target
    X, y, feature_names = select_features_and_target(pricing_df)
    
    # Step 4: Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    
    # Step 5: Train model
    model = train_xgboost_model(X_train, y_train)
    
    # Step 6: Evaluate model
    metrics, y_pred = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Step 7: Plot feature importance
    plot_feature_importance(model, feature_names)
    
    # Step 8: Plot actual vs predicted
    plot_actual_vs_predicted(y_test, y_pred)
    
    # Step 9: Save model
    save_model(model, feature_names, metrics)
    
    print("\n" + "=" * 70)
    print("✅ PRICE OPTIMIZATION MODEL TRAINING COMPLETE!")
    print("=" * 70)
    print("\n📁 Files created:")
    print("   • models/price_optimizer_model.pkl")
    print("   • models/price_optimizer_metadata.pkl")
    print("   • models/feature_importance.png")
    print("   • models/actual_vs_predicted.png")
    print("\n💰 This model can now suggest optimal prices to maximize revenue!")

# Run when script is executed
if __name__ == "__main__":
    main()