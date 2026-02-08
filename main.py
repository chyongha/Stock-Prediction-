from getdata import GetData
from sentiment import SentimentAnalyze
from train_data import StockPredictor
from visualize import Visualizer
import json

def main():
    ticker = input("Input the stock you are interested in (ex. TSLA) ")

    bot = GetData(ticker)
    stock_df = bot.get_stock_data()
    news_list = bot.get_news_headline()

    analyzer = SentimentAnalyze()
    enriched_news = []

    print("Begin Analyzing...")
    for item in news_list:
        headline = item[2]
        json_str = analyzer.headline_analyzer(headline)

        if json_str:
            data = json.loads(json_str)
            item.append(data['score'])
            item.append(data['reason'])
            enriched_news.append(item)
        
    predictor = StockPredictor()
    merged_df = predictor.preprocess(stock_df, enriched_news)

    print(f"Total row in the data: {len(merged_df)}")
    print(f"Lookback window: {predictor.lookback}")
    print(f"Expected Predction: {len(merged_df) - predictor.lookback}")
    print(f"Total Rows available for training: {len(merged_df)}")

    if len(merged_df) < 20:
        print("Warning! Not enough data..")
    model, X_tensor = predictor.build_model(merged_df)

    vis = Visualizer()
    vis.plot_result(model, X_tensor, predictor.scaler, merged_df, ticker)

if __name__ == "__main__":
    main()
    