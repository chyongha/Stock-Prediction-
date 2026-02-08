import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

class StockPredictor:
    def __init__(self, model_name= "lstm_v1.pth"):
        self.model_name = model_name
        self.scaler = MinMaxScaler(feature_range = (0, 1))
        self.model = None
        self.lookback = 3

    def preprocess(self, stock_df, news_list):
        print("Processing and merging data...")
        news_df = pd.DataFrame(news_list, columns = ['Date', 'Time', 'Text', 'Score', 'Reason'])

        def fix_date_format(date_str):
            if "Today" in date_str:
                return datetime.now().strftime("%Y-%m-%d")
            try:
                return datetime.strptime(date_str, "%b-%d-%y").strftime("%Y-%m-%d")
            except ValueError:
                pass

            if len(date_str.split('-')) == 3:
                return date_str  
            else:
                return f"{date_str}-{datetime.now().year}"
            
        news_df['Date'] = news_df['Date'].apply(fix_date_format)
        news_df['Date'] = pd.to_datetime(news_df['Date'], format='mixed')
        daily_sentiment = news_df.groupby('Date')['Score'].mean().reset_index()


        stock_df.index = pd.to_datetime(stock_df.index).tz_localize(None).normalize()
        merged_df = stock_df.merge(daily_sentiment, left_index = True, right_on = 'Date', how = 'left')
        merged_df['Score'] = merged_df['Score'].fillna(0.0)
        merged_df.to_csv("training_data.csv", index = False)

        print("Done saving the csv")
        return merged_df
    
    def create_sequence(self, data):
        dataset = data[['Close', 'Score']].values
        scaled_data = self.scaler.fit_transform(dataset)

        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i - self.lookback : i])
            y.append(scaled_data[i, 0])

        return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))
    
    def build_model(self, df, epochs = 100):
        print("Start training...")
        X, y = self.create_sequence(df)

        class LSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size = 2, hidden_size = 50, batch_first = True)
                self.linear = nn.Linear(50, 1)
            def forward(self, x):
                x, _ = self.lstm(x)
                return self.linear(x[:, -1, :])
        
        self.model = LSTM()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.01)

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.model(X)
            loss = criterion(output, y.unsqueeze(1))
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                print( f"Epoch {epoch}: Loss {loss.item(): .5f}")

        torch.save(self.model.state_dict(), self.model_name)
        print("Model Saved")
        return self.model, X