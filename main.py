# main.py

import pandas as pd
import logging
import os

from data_loading import load_data
from preprocessing import remove_capped_prices, fill_missing_values
from feature_engineering import engineer_features
from correlation_analysis import calculate_correlations
from visualizations import plot_correlation_heatmap
from model import HousingModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    """
    Main workflow for California Housing Analysis.
    Loads data, preprocesses, engineers features, analyzes correlations,
    visualizes, trains and evaluates a regression model.
    """
    try:
        df = load_data()
        logging.info("✅ Data loaded successfully.")
    except Exception as e:
        logging.error(f"❌ Error loading data: {e}")
        return

    logging.info(f"📊 Dataset shape: {df.shape}")
    logging.info("📋 Summary statistics:\n%s", df.describe())

    df = remove_capped_prices(df)
    logging.info(f"🧹 Capped prices removed. Remaining rows: {df.shape[0]}")

    logging.info("🔍 Missing values before:\n%s", df.isnull().sum())
    df = fill_missing_values(df)
    logging.info("✅ Missing values after:\n%s", df.isnull().sum())

    df = engineer_features(df)
    logging.info("🛠️ Engineered features (first 5 rows):\n%s", df[["rooms_per_household", "bedrooms_per_room", "population_per_household"]].head())

    correlations = calculate_correlations(df)
    logging.info("📈 Correlations with median_house_value:\n%s", correlations)

    # Save correlation heatmap
    plot_path = "figures/correlation_heatmap.png"
    plot_correlation_heatmap(df, save_path=plot_path)
    logging.info(f"🖼️ Correlation heatmap saved to {plot_path}")

    # Train and evaluate model
    model = HousingModel()
    model.train_model(df)
    mse, rmse = model.evaluate_model()
    logging.info("🤖 Final Model: Multiple Linear Regression with 4 features (train/test split)")
    logging.info(f"📉 Mean Squared Error (MSE): {mse:.2f}")
    logging.info(f"📉 Root Mean Squared Error (RMSE): {rmse:.2f}")

    print("\n✅ Analysis complete. See logs for details.")

if __name__ == "__main__":
    main()
