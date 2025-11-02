# src/shap_explainer.py
# SHAP (SHapley Additive exPlanations) for model interpretability

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def load_model_and_data():
    """
    Load the trained XGBoost model and test data
    Returns: model, X_test, y_test, feature_names
    """
    print("📂 Loading trained model and data...")
    
    # Load model
    model = joblib.load('models/price_optimizer_model.pkl')
    metadata = joblib.load('models/price_optimizer_metadata.pkl')
    feature_names = metadata['feature_names']
    
    print(f"✅ Loaded XGBoost model")
    print(f"   Features: {len(feature_names)}")
    
    # Load sales data
    df = pd.read_csv('data/processed/sales_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Recreate features (same as training)
    pricing_df = df.copy()
    
    # Same feature engineering as in train_price_model.py
    pricing_df['revenue_per_unit'] = pricing_df['revenue'] / pricing_df['quantity_sold']
    pricing_df['price'] = pricing_df['sale_price']
    pricing_df['discount'] = pricing_df['discount_pct']
    pricing_df['price_vs_competitor'] = pricing_df['sale_price'] - pricing_df['competitor_price']
    pricing_df['price_ratio_competitor'] = pricing_df['sale_price'] / (pricing_df['competitor_price'] + 0.01)
    pricing_df['price_percentile'] = pricing_df.groupby('product_id')['sale_price'].rank(pct=True)
    pricing_df['category_encoded'] = pricing_df['category'].astype('category').cat.codes
    pricing_df['demand'] = pricing_df['quantity_sold']
    pricing_df['demand_ma7'] = pricing_df.groupby('product_id')['quantity_sold'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    pricing_df['price_change'] = pricing_df.groupby('product_id')['sale_price'].diff().fillna(0)
    
    # Select features
    df_clean = pricing_df[feature_names + ['revenue']].dropna()
    X = df_clean[feature_names]
    y = df_clean['revenue']
    
    # Take a sample for SHAP (SHAP is slow on large datasets)
    # Sample 1000 random rows
    sample_size = min(1000, len(X))
    sample_indices = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X.iloc[sample_indices]
    y_sample = y.iloc[sample_indices]
    
    print(f"✅ Loaded {len(X_sample)} samples for SHAP analysis")
    
    return model, X_sample, y_sample, feature_names

def create_shap_explainer(model, X_sample):
    """
    Create SHAP explainer object
    
    Args:
        model: Trained XGBoost model
        X_sample: Sample data for background
    Returns: SHAP explainer and values
    """
    print("\n🧠 Creating SHAP explainer...")
    print("   This may take 2-3 minutes for 1000 samples... ⏳")
    
    # Create TreeExplainer (optimized for XGBoost/tree models)
    # TreeExplainer is MUCH faster than KernelExplainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for all samples
    # SHAP values show how much each feature contributed to each prediction
    shap_values = explainer.shap_values(X_sample)
    
    print("✅ SHAP values calculated!")
    
    return explainer, shap_values

def plot_shap_summary(shap_values, X_sample, save_path='models/shap_summary_plot.png'):
    """
    Create SHAP summary plot
    Shows which features are most important overall
    
    Args:
        shap_values: SHAP values array
        X_sample: Sample data
        save_path: Where to save plot
    """
    print("\n📊 Creating SHAP summary plot...")
    
    # Create summary plot
    plt.figure(figsize=(12, 8))
    
    # Summary plot shows:
    # - Y-axis: Features ranked by importance
    # - X-axis: SHAP value (impact on prediction)
    # - Color: Feature value (red=high, blue=low)
    # - Each dot is one prediction
    
    shap.summary_plot(
        shap_values, 
        X_sample, 
        plot_type="dot",
        show=False
    )
    
    plt.title('SHAP Summary Plot - Feature Impact on Revenue Predictions\n' + 
              '(Red=High feature value, Blue=Low feature value)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ SHAP summary plot saved to: {save_path}")
    
    plt.show()

def plot_shap_bar(shap_values, X_sample, save_path='models/shap_bar_plot.png'):
    """
    Create SHAP bar plot
    Shows average absolute impact of each feature
    
    Args:
        shap_values: SHAP values array
        X_sample: Sample data
        save_path: Where to save plot
    """
    print("\n📊 Creating SHAP bar plot...")
    
    plt.figure(figsize=(10, 8))
    
    # Bar plot shows mean absolute SHAP value
    # = average importance of each feature
    shap.summary_plot(
        shap_values, 
        X_sample, 
        plot_type="bar",
        show=False
    )
    
    plt.title('SHAP Bar Plot - Average Feature Importance\n' + 
              '(Mean absolute impact on revenue)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ SHAP bar plot saved to: {save_path}")
    
    plt.show()

def plot_shap_waterfall(explainer, shap_values, X_sample, index=0, 
                        save_path='models/shap_waterfall_plot.png'):
    """
    Create SHAP waterfall plot for a single prediction
    Shows step-by-step how features contributed to ONE specific prediction
    
    Args:
        explainer: SHAP explainer object
        shap_values: SHAP values array
        X_sample: Sample data
        index: Which sample to explain (default: first one)
        save_path: Where to save plot
    """
    print(f"\n📊 Creating SHAP waterfall plot for sample #{index}...")
    
    plt.figure(figsize=(12, 8))
    
    # Create explanation object for one sample
    explanation = shap.Explanation(
        values=shap_values[index],
        base_values=explainer.expected_value,
        data=X_sample.iloc[index],
        feature_names=X_sample.columns.tolist()
    )
    
    # Waterfall plot shows:
    # - Start with base value (average prediction)
    # - Each bar shows how one feature pushed prediction up/down
    # - End with final prediction
    
    shap.waterfall_plot(explanation, show=False)
    
    plt.title(f'SHAP Waterfall Plot - Single Prediction Breakdown (Sample #{index})\n' + 
              '(How each feature contributed to this specific prediction)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ SHAP waterfall plot saved to: {save_path}")
    
    plt.show()

def plot_shap_force(explainer, shap_values, X_sample, index=0, 
                    save_path='models/shap_force_plot.html'):
    """
    Create SHAP force plot (interactive HTML)
    Visualizes how features push prediction higher or lower
    
    Args:
        explainer: SHAP explainer object
        shap_values: SHAP values array
        X_sample: Sample data
        index: Which sample to explain
        save_path: Where to save HTML file
    """
    print(f"\n📊 Creating SHAP force plot for sample #{index}...")
    
    # Force plot shows:
    # - Red bars: Features pushing prediction HIGHER
    # - Blue bars: Features pushing prediction LOWER
    # - Width of bar = strength of impact
    
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[index],
        X_sample.iloc[index],
        matplotlib=False  # Use interactive D3.js version
    )
    
    # Save as HTML (interactive!)
    shap.save_html(save_path, force_plot)
    print(f"✅ SHAP force plot saved to: {save_path}")
    print(f"   Open this file in a browser to see interactive visualization!")

def plot_shap_dependence(shap_values, X_sample, feature='price', 
                         save_path='models/shap_dependence_price.png'):
    """
    Create SHAP dependence plot
    Shows how one feature's value affects predictions
    
    Args:
        shap_values: SHAP values array
        X_sample: Sample data
        feature: Which feature to analyze
        save_path: Where to save plot
    """
    print(f"\n📊 Creating SHAP dependence plot for '{feature}'...")
    
    plt.figure(figsize=(10, 6))
    
    # Dependence plot shows:
    # - X-axis: Feature value (e.g., price)
    # - Y-axis: SHAP value (impact on prediction)
    # - Color: Interaction with another feature
    # This shows non-linear relationships!
    
    shap.dependence_plot(
        feature,
        shap_values,
        X_sample,
        show=False
    )
    
    plt.title(f'SHAP Dependence Plot - How "{feature}" Affects Revenue\n' + 
              '(Shows non-linear relationships)', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ SHAP dependence plot saved to: {save_path}")
    
    plt.show()

def analyze_sample_predictions(model, explainer, shap_values, X_sample, y_sample, num_samples=3):
    """
    Analyze and explain specific predictions
    
    Args:
        model: Trained model
        explainer: SHAP explainer
        shap_values: SHAP values
        X_sample: Sample features
        y_sample: Sample targets
        num_samples: How many samples to analyze
    """
    print("\n🔍 ANALYZING SAMPLE PREDICTIONS")
    print("=" * 70)
    
    # Make predictions
    predictions = model.predict(X_sample)
    
    # Analyze first few samples
    for i in range(min(num_samples, len(X_sample))):
        actual = y_sample.iloc[i]
        predicted = predictions[i]
        error = actual - predicted
        error_pct = (error / actual) * 100
        
        print(f"\n📌 Sample #{i}:")
        print(f"   Actual Revenue:    ₹{actual:.2f}")
        print(f"   Predicted Revenue: ₹{predicted:.2f}")
        print(f"   Error:             ₹{error:.2f} ({error_pct:.1f}%)")
        print(f"   Base Value:        ₹{explainer.expected_value:.2f}")
        
        # Get top 3 positive and negative contributors
        feature_contributions = pd.DataFrame({
            'Feature': X_sample.columns,
            'Value': X_sample.iloc[i].values,
            'SHAP': shap_values[i]
        }).sort_values('SHAP', key=abs, ascending=False)
        
        print("\n   Top 3 Features Increasing Revenue:")
        for idx, row in feature_contributions[feature_contributions['SHAP'] > 0].head(3).iterrows():
            print(f"      {row['Feature']}: {row['Value']:.2f} → +₹{row['SHAP']:.2f}")
        
        print("\n   Top 3 Features Decreasing Revenue:")
        for idx, row in feature_contributions[feature_contributions['SHAP'] < 0].head(3).iterrows():
            print(f"      {row['Feature']}: {row['Value']:.2f} → ₹{row['SHAP']:.2f}")

def main():
    """
    Main function to run SHAP analysis
    """
    print("=" * 70)
    print("🚀 SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 70)
    print("💡 Making AI predictions transparent and trustworthy!")
    print("=" * 70)
    
    # Step 1: Load model and data
    model, X_sample, y_sample, feature_names = load_model_and_data()
    
    # Step 2: Create SHAP explainer
    explainer, shap_values = create_shap_explainer(model, X_sample)
    
    # Step 3: Create SHAP summary plot (most important!)
    plot_shap_summary(shap_values, X_sample)
    
    # Step 4: Create SHAP bar plot
    plot_shap_bar(shap_values, X_sample)
    
    # Step 5: Create waterfall plot (single prediction)
    plot_shap_waterfall(explainer, shap_values, X_sample, index=0)
    
    # Step 6: Create force plot (interactive HTML)
    plot_shap_force(explainer, shap_values, X_sample, index=0)
    
    # Step 7: Create dependence plot for price
    plot_shap_dependence(shap_values, X_sample, feature='price')
    
    # Step 8: Create dependence plot for discount
    plot_shap_dependence(shap_values, X_sample, feature='discount', 
                         save_path='models/shap_dependence_discount.png')
    
    # Step 9: Analyze specific predictions
    analyze_sample_predictions(model, explainer, shap_values, X_sample, y_sample, num_samples=3)
    
    print("\n" + "=" * 70)
    print("✅ SHAP EXPLAINABILITY ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\n📁 Files created:")
    print("   • models/shap_summary_plot.png (Overall feature importance)")
    print("   • models/shap_bar_plot.png (Average feature impact)")
    print("   • models/shap_waterfall_plot.png (Single prediction breakdown)")
    print("   • models/shap_force_plot.html (Interactive visualization)")
    print("   • models/shap_dependence_price.png (Price impact curve)")
    print("   • models/shap_dependence_discount.png (Discount impact curve)")
    print("\n💡 SHAP makes your model transparent and trustworthy!")
    print("   Now stakeholders can understand WHY the AI makes each decision.")

# Run when script is executed
if __name__ == "__main__":
    main()