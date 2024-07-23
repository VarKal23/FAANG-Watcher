# FAANG Stock Price Prediction App

This repository contains a fullstack application designed to compare the closing price outputs of FAANG stocks using various deep learning and machine learning models. The app integrates multiple models to provide real-time stock analysis and recommendations.

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
    git clone https://github.com/yourusername/FAANG-stock-prediction-app.git
    cd FAANG-stock-prediction-app
    ```

2. **Backend Setup:**

    ```bash
    cd backend
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3. **Frontend Setup:**

    ```bash
    cd frontend
    npm install
    npm start
    ```

### Configuration

- **Finnhub API Key**: Add your Finnhub API key to the `.env` file in the backend directory.

    ```env
    FINNHUB_API_KEY=your_finnhub_api_key
    ```

## Usage

- **Start Backend Server**: 

    ```bash
    cd backend
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    python app.py
    ```

- **Start Frontend Server**:

    ```bash
    cd frontend
    npm start
    ```

## Models

### LSTM (Long Short-Term Memory)

- **Description**: Custom LSTM RNN designed with a TBTT (Truncated Backpropagation Through Time) length of 90 to optimize model for estimating quarterly trends.
- **Training Data**: Trained on 5 years of FAANG historical data including Prices, RSI, Volume, MACD, and MA50 obtained through YFinance.
- **Notebook**: 

### Linear Regression

- **Description**: Simple linear regression model for stock price prediction.
- **Usage**: Details on Linear Regression model. *(To be added)*

### ARIMA (AutoRegressive Integrated Moving Average)

- **Description**: Time series forecasting model for stock price prediction.
- **Usage**: Details on ARIMA model. *(To be added)*

