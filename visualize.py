import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd

class Visualizer:
    def plot_result(self, model, X_tensor, scaler, original_df, ticker):
        print("📊 Generating Custom Chart...")
        
        model.eval()
        with torch.no_grad():
            predicted = model(X_tensor).numpy()

        dummy = np.zeros((len(predicted), 2))
        dummy[:, 0] = predicted.flatten()
        real_predicted_price = scaler.inverse_transform(dummy)[:, 0]

        lookback = len(original_df) - len(real_predicted_price)
        
        if 'Date' in original_df.columns:
            plot_dates = pd.to_datetime(original_df['Date'][lookback:])
        else:
            plot_dates = original_df.index[lookback:]

        print(f"DEBUG: Original Rows: {len(original_df)}, Predicted: {len(real_predicted_price)}")
        print(f"DEBUG: Calculated Lookback: {lookback}")

        plt.style.use('dark_background') 
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # --- TOP CHART: PRICE ---
        ax1.grid(True, which='major', color='#444444', linestyle='--', alpha=0.3)
        
        ax1.plot(plot_dates, original_df['Close'][lookback:], 
                 label='Actual Price', color='#00d4ff', linewidth=2, alpha=0.9)
        
        ax1.plot(plot_dates, real_predicted_price, 
                 label='AI Prediction', color='#ff0055', linestyle='--', linewidth=2)
        
        ax1.set_ylabel('Stock Price ($)', fontsize=12, color='white')
        ax1.legend(loc='upper left', facecolor='#222222', edgecolor='white')
        ax1.set_title(f"Stock Price vs AI Prediction", fontsize=16, color='white', pad=10)

        sentiment_scores = original_df['Score'][lookback:]
        
        colors = ['#00ff00' if s >= 0 else '#ff0000' for s in sentiment_scores]
        
        ax2.bar(plot_dates, sentiment_scores, alpha=0.8, color=colors, width=0.6)
        
        ax2.axhline(0, color='white', linewidth=1, alpha=0.5) # Zero line
        ax2.set_ylabel('News Sentiment', fontsize=12, color='white')
        ax2.set_ylim(-1.1, 1.1) 
        ax2.grid(True, axis='y', color='#444444', linestyle='--', alpha=0.3)

        plt.xlabel('Date', fontsize=12, color='white')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.05) 
        plt.show()