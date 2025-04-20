import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def fetch_stock_data(symbol, period='1y'):
    """Fetch stock data using yfinance"""
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    return df

def prepare_data(df):
    """Prepare data for prediction"""
    # Calculate technical indicators
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    
    # Create features
    df['Target'] = df['Close'].shift(-1)  # Tomorrow's price
    df = df.dropna()
    
    # Prepare features
    features = ['Close', 'Volume', 'SMA20', 'SMA50']
    X = df[features]
    y = df['Target']
    
    return X, y

def predict_future_prices(model, last_data, num_days):
    """Predict stock prices for the next n days"""
    predictions = []
    current_data = last_data.copy()
    
    for _ in range(num_days):
        next_price = model.predict(current_data.reshape(1, -1))[0]
        predictions.append(next_price)
        current_data[0] = next_price
        current_data[2] = (current_data[2] * 19 + next_price) / 20
        current_data[3] = (current_data[3] * 49 + next_price) / 50
    
    return predictions

class StockPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Price Prediction")
        self.root.geometry("1200x800")

        # Create main frame
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create input frame
        self.input_frame = ttk.Frame(self.main_frame)
        self.input_frame.pack(fill=tk.X, padx=5, pady=5)

        # Stock symbol input
        ttk.Label(self.input_frame, text="Stock Symbol:").pack(side=tk.LEFT, padx=5)
        self.symbol_var = tk.StringVar(value="AAPL")
        self.symbol_entry = ttk.Entry(self.input_frame, textvariable=self.symbol_var)
        self.symbol_entry.pack(side=tk.LEFT, padx=5)

        # Days input
        ttk.Label(self.input_frame, text="Days to Predict:").pack(side=tk.LEFT, padx=5)
        self.days_var = tk.StringVar(value="30")
        self.days_entry = ttk.Entry(self.input_frame, textvariable=self.days_var)
        self.days_entry.pack(side=tk.LEFT, padx=5)

        # Predict button
        self.predict_button = ttk.Button(self.input_frame, text="Predict", command=self.make_prediction)
        self.predict_button.pack(side=tk.LEFT, padx=5)

        # Create scrollable text area with scrollbar on the left
        self.text_frame = ttk.Frame(self.main_frame)
        self.text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create scrollbar on the left
        self.scrollbar = ttk.Scrollbar(self.text_frame)
        self.scrollbar.pack(side=tk.LEFT, fill=tk.Y)

        # Create text widget
        self.text_area = tk.Text(self.text_frame, wrap=tk.WORD, yscrollcommand=self.scrollbar.set)
        self.text_area.pack(fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.text_area.yview)

        # Create frame for matplotlib figures
        self.fig_frame = ttk.Frame(self.main_frame)
        self.fig_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def make_prediction(self):
        symbol = self.symbol_var.get().upper()
        try:
            days = int(self.days_var.get())
        except ValueError:
            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(tk.END, "Please enter a valid number of days")
            return

        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, f"Fetching data for {symbol}...\n\n")
        self.root.update()

        try:
            # Fetch and prepare data
            df = fetch_stock_data(symbol)
            X, y = prepare_data(df)
            
            # Split and train
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Calculate accuracy
            accuracy = model.score(X_test, y_test)
            self.text_area.insert(tk.END, f"Model R² Score: {accuracy:.4f}\n\n")

            # Predict future prices
            latest_data = X.iloc[-1].values
            future_predictions = predict_future_prices(model, latest_data, days)
            last_date = df.index[-1]
            future_dates = [last_date + timedelta(days=i+1) for i in range(days)]

            # Display predictions
            self.text_area.insert(tk.END, f"Predicted prices for {symbol} over next {days} days:\n\n")
            for date, price in zip(future_dates, future_predictions):
                self.text_area.insert(tk.END, f"{date.date()}: ${price:.2f}\n")

            # Clear previous plots
            for widget in self.fig_frame.winfo_children():
                widget.destroy()

            # Create new figure for historical and future predictions
            fig = Figure(figsize=(12, 6))
            ax = fig.add_subplot(111)
            
            # Plot historical data
            ax.plot(df.index[-30:], df['Close'][-30:], label='Historical Prices')
            ax.plot(future_dates, future_predictions, 'r--', label='Future Predictions')
            ax.set_title(f'{symbol} Stock Price Prediction - Next {days} Days')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            
            # Rotate x-axis labels for better readability
            fig.autofmt_xdate()

            # Add the plot to the GUI
            canvas = FigureCanvasTkAgg(fig, master=self.fig_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(tk.END, f"Error: {str(e)}")

def main():
    root = tk.Tk()
    app = StockPredictionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()