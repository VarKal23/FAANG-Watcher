import pandas as pd
import os
import datetime
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
from datetime import datetime
from historical_data import get_90_stock_data


# LSTM Model Architecture
class FaangRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(FaangRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = 2

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            self.num_layers,
            batch_first=True,
            dropout=.1
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        hidden_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        cell_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        output, (h, c) = self.lstm(x, (hidden_state.detach(), cell_state.detach()))

        last_output = output[:, -1, :]
        linear_output = self.fc(last_output)

        return linear_output


def lstm_future_predictions(ticker, pred_days, end_date):
    '''
    stock: string; stock ticker
    start, end: string; YYYY-MM-DD
    pred_days: number of desired predictions

    returns the mse, actual adjusted closing prices, and predicted adusted closing prices
    '''
    pred_days = int(pred_days)
    input_size = 5
    output_size = 5
    hidden_size = 48

    model = FaangRNN(input_size, output_size, hidden_size)
    print(os.getcwd())
    # load the saved model and optimizer
    checkpoint = torch.load('pretrained_models/FAANG_RNN.pth')
    
    model.load_state_dict(checkpoint['model_state_dict'])

    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Fitting scaler on the combined data
    data = []
    for stock in ['META', 'AAPL', 'AMZN', 'NFLX', 'GOOG']:
        df = get_90_stock_data(stock, end_date, cols=['Adj Close', 'Volume', 'RSI', 'MA50', 'MACD'])
        data.append(df)
    # combine data into a single df for fitting the scaler
    combined_data = pd.concat(data)
    print(combined_data)
    # fit the scaler on the combined data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(combined_data[['Adj Close', 'Volume', 'RSI', 'MA50', 'MACD']])

    cols = ['Adj Close', 'Volume', 'RSI', 'MA50', 'MACD']

    # get the test data for the last 90 days
    print("The stock is: ", ticker)
    test_data = get_90_stock_data(stock=ticker, end_date=end_date, cols=cols)
    test_data[cols] = scaler.transform(test_data[['Adj Close', 'Volume', 'RSI', 'MA50', 'MACD']])

    test_tensor = torch.tensor(test_data.values, dtype=torch.float32)

    # get predicted adj close from model
    input_seq = test_tensor.unsqueeze(0)
    last_pred = model(input_seq)
    last_pred_unscaled = scaler.inverse_transform(last_pred.detach().numpy())
    predicted_seq = [last_pred_unscaled[0, 0].item()]

    with torch.no_grad():
        for _ in range(pred_days - 1):
            new_record = torch.tensor([[last_pred[0, 0].item(), last_pred[0, 1].item(), last_pred[0, 2].item(), last_pred[0, 3].item(), last_pred[0, 4].item()]], dtype=torch.float32)
            input_seq = torch.cat((input_seq[:, :, :], new_record.unsqueeze(0)), dim=1)
            pred = model(input_seq)
            last_pred = pred
            last_pred_unscaled = scaler.inverse_transform(last_pred.detach().numpy())
            predicted_seq.append(last_pred_unscaled[0, 0].item())

    predicted_tensor = torch.tensor(predicted_seq, dtype=torch.float32)

    return {'Adj Close': predicted_tensor.tolist(), 'Date': trading_days(pred_days, start_date=pd.to_datetime(end_date))}


def trading_days(num_days, start_date=datetime.now()):
    '''Returns a list of trading days for the given number of days starting from the given date.'''
    import pandas_market_calendars as mcal
    from datetime import datetime, timedelta

    market = mcal.get_calendar('XNYS') # For the New York Stock Exchange (NYSE)
    today = start_date  

    trading_days = []
    current_date = today

    while len(trading_days) < num_days:
        schedule = market.schedule(start_date=current_date + timedelta(days=1), end_date=current_date + pd.DateOffset(days=num_days) + timedelta(days=1)) 
        trading_days.extend(schedule.index.strftime('%Y-%m-%d'))
        current_date += pd.DateOffset(days=num_days)

    trading_days = trading_days[:num_days]
    return trading_days
    

