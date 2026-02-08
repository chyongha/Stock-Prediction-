import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import yfinance as yf

class GetData:
    def __init__(self, ticker):
        self.ticker = ticker

    def get_stock_data(self, days = 365):
        print(f"Fetching stock data for {self.ticker}...")
        stock = yf.Ticker(self.ticker)

        start_date = datetime.now() - timedelta(days = days + 10)
        df = stock.history(start = start_date, end = datetime.now())
        return df[["Close", "Volume"]]

    def get_news_headline(self):
        print(f"Scraping news for {self.ticker}")
        url = f"https://finviz.com/quote.ashx?t={self.ticker}"
        headers = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

        try:
            req = requests.get(url, headers = headers)
            soup = BeautifulSoup(req.content, 'html.parser')
            news_table = soup.find(id = 'news-table')

            if not news_table:
                print("No news table found")
                return []
            
            parsed_news = []
            current_date = ""
            for x in news_table.find_all('tr'):
                if x.a:
                    text = x.a.get_text()
                    date_scrape = x.td.text.split()

                    if len(date_scrape) == 1:
                        time = date_scrape[0]
                    else:
                        current_date = date_scrape[0]
                        time = date_scrape[1]

                        parsed_news.append([current_date, time, text])
                
            return parsed_news
        except Exception as es:
            print(f"Failed scraping due to {es}")
            return []