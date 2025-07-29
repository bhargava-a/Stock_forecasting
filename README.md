# Stock Price Prediction Pipeline

## What This Project Does
This project predicts stock prices for BEL (Bharat Electronics Limited) using machine learning. It takes historical stock data and tries to predict what the stock price will be tomorrow.

## Files
- `pipeline.ipynb` - Main notebook with all the code
- `BEL_Full_Stock_Data_2002_2025.csv` - Stock data file
- `bel_xgb_model.pkl` - Saved trained model

## How It Works (Step by Step)

### 1. Data Loading & Cleaning
- Loads stock data from CSV file
- Removes bad data and fixes data types
- Sorts data by date

### 2. Exploratory Data Analysis (EDA)
- Shows basic statistics of the data
- Creates plots to understand price trends
- Shows relationships between different features

### 3. Feature Engineering
Creates new features from existing data:
- **Price Change**: How much price changed each day
- **Daily Return**: Percentage change in price
- **Rolling Averages**: Average price over last 5, 10 days
- **Volatility**: How much price varies over time
- **RSI**: Technical indicator showing if stock is overbought/oversold
- **Lag Features**: Previous day's prices (1-5 days back)

### 4. Target Variable
- **Target**: Next day's closing price
- This is what we want to predict

### 5. Model Training
Uses **XGBoost** (like Random Forest but more advanced):
- **Training Set**: 80% of data (older dates)
- **Test Set**: 20% of data (newer dates)
- **Hyperparameter Tuning**: Finds best settings for the model

### 6. Model Evaluation
Measures how well the model predicts:
- **MAE**: Average prediction error
- **RMSE**: Square root of average squared error
- **R²**: How much of the price variation the model explains

## Results
- **R² Score**: ~0.80 (80% accuracy)
- **RMSE**: ~42 points
- Model can predict stock prices with reasonable accuracy

## Key Concepts Used
- **Time Series Split**: Unlike regular train_test_split, we use time-based splitting (older data for training, newer for testing)
- **Feature Engineering**: Creating new useful features from existing data
- **Hyperparameter Tuning**: Finding the best settings for the model
- **Cross-Validation**: Testing model on different parts of data

## Libraries Used
- **pandas**: Data manipulation
- **numpy**: Math operations
- **matplotlib/seaborn**: Plotting
- **scikit-learn**: Machine learning tools
- **xgboost**: The main prediction model
- **shap**: Model explanation

## How to Run
1. Open `pipeline.ipynb` in Jupyter Notebook
2. Run all cells in order
3. The model will train and show results

## Notes
- Stock prediction is inherently difficult due to market randomness
- This model uses only historical price data
- Adding news, sentiment, or economic data could improve accuracy
- Results are for educational purposes, not financial advice 