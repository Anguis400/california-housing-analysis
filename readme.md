# 🏡 California Housing Analysis

Exploratory Data Analysis (EDA) and Multiple Linear Regression project using the California Housing dataset.

---

## 📁 Project Structure

- `main.py`: Main script to execute the full pipeline.
- `data_loading.py`: Loads the dataset.
- `preprocessing.py`: Handles null values and data cleaning.
- `feature_engineering.py`: Creates new variables.
- `correlation_analysis.py`: Performs correlation analysis.
- `visualizations.py`: Generates visual charts.
- `model.py`: Trains and evaluates the regression model.
- `figures/`: Automatically saved plots and visualizations.

---

## ✅ Features

- Variable engineering
- Correlation heatmap
- Multiple Linear Regression model
- Modular, clean, and well-commented code
- Step-by-step logging messages

---

## 📊 Model Results

- **Model**: Multiple Linear Regression  
- **Selected features**:  
  `median_income`, `housing_median_age`, `rooms_per_household`, `bedrooms_per_room`  
- **MSE**: ~4.99e9  
- **RMSE**: ~70,642

---

## ▶️ How to Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt

