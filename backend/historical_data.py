import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

stocks = ["AAPL", "AMZN", "GOOGL", "META", "NFLX"]

def get_date_from_duration(stock, duration):
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=duration)

    return stock.history(start=start_date, end=end_date)['Close']

# Function to get stock data
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1d')  # Adjust period as needed
        volume = data['Volume'].iloc[-1]  # Volume for the latest trading day
        price = data['Close'].iloc[-1]  # Price at the end of the trading day

        # Fetching historical data for 7 days, 30 days, 1 year
        hist_7d = get_date_from_duration(stock, 7)
        hist_30d = get_date_from_duration(stock, 30)
        hist_1y = get_date_from_duration(stock, 365)

        # Calculate today's change as a percentage
        today_open = data['Open'].iloc[-1]
        today_change = ((price - today_open) / today_open) * 100

        return {
            'Ticker': ticker,
            'Volume': volume,
            'Price': round(price, 2),
            '7D': round(hist_7d[0], 2),  # Value of stock at the end of 7 days
            '30D': round(hist_30d[0], 2),  # Value of stock at the end of 30 days
            '1Y': round(hist_1y[0], 2),  # Value of stock at the end of 1 year
            'Today Change': round(today_change, 2)  # Round to 2 decimal places for percentage
        }
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def get_latest_stock_data():
    # Fetching data for all stocks
    stock_data = []
    for stock in stocks:
        data = get_stock_data(stock)
        if data:
            stock_data.append(data)

    # Creating DataFrame
    df = pd.DataFrame(stock_data)

    # Sort by volume and return top 10 stocks
    top_10_stocks = df.sort_values(by='Volume', ascending=False).head(10)
    return top_10_stocks


# Get YFinance stock data for past 90 days, for use with LSTM model
def get_90_stock_data(stock, end_date, cols=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'RSI', 'MA50', 'EMA12', 'EMA26', 'MACD']):

    # Calculate end date as today if not provided
    end = pd.to_datetime(end_date)
    print(end)
    #TODO: Check for off by one error
    # Calculate start date as 90 days before the end date
    start = (end - timedelta(days=90)).strftime('%Y-%m-%d')
    print(start)
    end = end.strftime('%Y-%m-%d')

    # retrieve stock data starting 100 days early for relative strength index (RSI) and 50-day moving average (MA50) calculations
    rsi_start = datetime.strptime(start, '%Y-%m-%d') - timedelta(days=100)
    rsi_start = rsi_start.strftime('%Y-%m-%d')

    df = yf.download(stock, start=rsi_start, end=end)

    # calculate MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']

    # calculate MA50
    df['MA50'] = df['Close'].rolling(window=50).mean()

    # calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (gain / loss + 1))

    # remove data before start that was retrieved just for the RSI calculation
    df = df.loc[start:]

    # only return specified cols
    return df[cols]
