# FAANG Stock Price Prediction App

This repository contains a fullstack application designed to compare the closing price outputs of different FAANG stocks using various deep learning and machine learning models. Comparing these outputs leads us to draw interesting conclusions about the efficacy of different ML/DL models on a controlled group of stocks.

## Features

- **Model Integration**: Incorporates LSTM, Linear Regression, and ARIMA models.
- **Frontend**: User interface built with React to display model input/output.
- **Backend**: Flask-based server for model integration and data handling.
- **Real-Time Data**: Displays live buy/sell/hold recommendations, stock metrics, news, and more using the Finnhub client.
- **Custom LSTM RNN**: Engineered an LSTM RNN with TBTT length of 90 for optimal quarterly trend estimation.
- **Data Sources**: Trained on 5 years of historical data including Prices, RSI, Volume, MACD, and MA50 obtained through YFinance and feature engineering.

## Getting Started

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/VarKal23/FAANG-Watcher.git
    cd FAANG-stock-prediction-app
    ```

2. **Backend Setup:**

    ```bash
    cd backend
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    flask --app server.py run 
    ```

3. **Frontend Setup:**

    ```bash
    cd frontend
    npm install
    npm run dev
    ```

### Configuration

- **Fronted env file**: Add a finnhub client api key to a .env file in the frontend directory.

    ```env
    REACT_APP_API_KEY=your_finnhub_api_key
    ```

## Models

### LSTM (Long Short-Term Memory)

- **Description**: Custom LSTM RNN designed with a TBTT (Truncated Backpropagation Through Time) length of 90 to optimize model for estimating quarterly trends.
- **Training Data**: Trained on 5 years of FAANG historical data including Prices, RSI, Volume, MACD, and MA50 obtained through YFinance.
- **Notebook**: *(To be added)*

### Linear Regression

- **Description**: Simple linear regression model for stock price prediction.
- **Usage**: Details on Linear Regression model. *(To be added)*

### ARIMA (AutoRegressive Integrated Moving Average)

- **Description**: Time series forecasting model for stock price prediction.
- **Usage**: Details on ARIMA model. *(To be added)*

