# %% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import requests
import json
import time
import random
import re
import datetime
import pytz
import pickle


preprocessed_data = pd.read_csv('cleaned_preprocessed.csv')

# print(preprocessed_data.head())
# print(preprocessed_data.info())
# print(preprocessed_data.describe())
# print(preprocessed_data.columns)
# print(preprocessed_data.shape)
# print(preprocessed_data.dtypes)
# print(preprocessed_data.isnull().sum())
print(preprocessed_data)

print(f"-----------------")
# print(preprocessed_data.date.tail(50))


# %%
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import requests
import json
import time
import random
import re
import datetime
import pytz
import pickle
import gs_quant 
import yfinance as yf
import pandas_datareader as pdr
import datetime
from datetime import datetime, timedelta


start_date = datetime.utcnow() - timedelta(days=3)
end_date = datetime.utcnow()
symbol = "AAPL"

df = yf.download(symbol, start=start_date, end=end_date, interval='1m',)
print(df)
plt.figure(figsize=(10, 5))
plt.plot(df.index, df.Close)
plt.show()


# %%
#!pip install yfinance
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

# Download 1-minute data for the last 2 days
# start_date = datetime.utcnow() - timedelta(days=2)
# end_date = datetime.utcnow()
length = 8
start_date = datetime.utcnow() - timedelta(days=length)
end_date = datetime.utcnow()
symbol = "MMM"

df = yf.download(symbol, start=start_date, end=end_date, interval='1m', progress=False)
df = df.tz_convert('US/Eastern')

# Filter for market open hours (9:30 to 16:00)
is_open = (df.index.time >= datetime.strptime("0:00", "%H:%M").time()) & \
          (df.index.time <= datetime.strptime("20:00", "%H:%M").time())
df_open = df[is_open]

# Create a continuous x-axis: "market minute index"
df_open = df_open.copy()
df_open['day'] = df_open.index.date

# Count minutes since the start of the first day (stitching all open periods together)
minute_offsets = []
offset = 0
for day, group in df_open.groupby('day'):
    n = len(group)
    minute_offsets.extend(range(offset, offset + n))
    offset += n

df_open['market_minute'] = minute_offsets

# Plot
plt.figure(figsize=(14, 6))
plt.plot(df_open['market_minute'], df_open['Close'], color='b')

# Optionally, add vertical lines to indicate day changes
for day, group in df_open.groupby('day'):
    plt.axvline(group['market_minute'].iloc[0], color='gray', linestyle='--', alpha=0.5)
    plt.text(group['market_minute'].iloc[0], plt.ylim()[1], str(day), va='bottom', ha='left', fontsize=8, color='gray')

plt.xlabel("Market Minutes (stitched, no overnight gaps)")
plt.ylabel("AAPL Close Price")
plt.title("AAPL Intraday Prices (Market Hours Only, No Gaps Between Days)")
plt.tight_layout()
plt.show()

print(len(df_open))

# %%

631587/2302 

# %%

import os 
import pandas as pd
text_file = open("all_etf.txt", "r")
etf_ticker = text_file.read()
text_file.close()
etf_ticker = etf_ticker.split("\n")
etf_ticker = [ticker.strip() for ticker in etf_ticker if ticker.strip()]

# Extract only the ticker symbols (first column before tab)
ticker_symbols = []
for line in etf_ticker:
    if line.strip():  # Skip empty lines
        symbol = line.split("\t")[0].strip()  # Split by tab and take first part
        ticker_symbols.append(symbol)
        print(symbol)

print(f"\nTotal ETF symbols extracted: {len(ticker_symbols)}")
# %%

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from collections import Counter
from math import ceil, floor 
import pandas as pd
import pandas_ta as pdta
import config
import time
import schedule
from datetime import datetime
from datetime import date
from datetime import timedelta
import numpy as np
import statsmodels.api as sm
import scipy as sp
from scipy.stats import theilslopes , linregress
from scipy.ndimage import gaussian_filter1d , generic_filter1d
import scipy.signal as signal
import scipy.interpolate as si
from scipy.signal import find_peaks
from scipy import stats
import pywt
import mplcursors
import pickle
import ta as ta
import pytz
import datetime
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.trading.requests import MarketOrderRequest ,LimitOrderRequest, ClosePositionRequest
from alpaca.data.requests import StockLatestQuoteRequest, CryptoLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit
from IPython.display import display, clear_output
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas_ta as pdta
import config
import os
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderSide,OrderClass,OrderType, QueryOrderStatus, OrderStatus
from sklearn.preprocessing import MinMaxScaler

# Initialize Alpaca clients
count = 0
alt_pkl_file_path = 0
if alt_pkl_file_path == 0:
    trading_client = TradingClient(config.alpaca_paper_api_key, config.alpaca_paper_secret_key, paper=True)
    stock_client = StockHistoricalDataClient(config.alpaca_paper_api_key, config.alpaca_paper_secret_key)
    api = REST(config.alpaca_paper_api_key, config.alpaca_paper_secret_key, base_url=config.alpaca_paper_endpoint)
elif alt_pkl_file_path == 1:
    trading_client = TradingClient(config.alpaca_paper_counter_api_key, config.alpaca_paper_counter_secret_key, paper=True)
    stock_client = StockHistoricalDataClient(config.alpaca_paper_counter_api_key, config.alpaca_paper_counter_secret_key)
    api = REST(config.alpaca_paper_counter_api_key, config.alpaca_paper_counter_secret_key, base_url=config.alpaca_paper_counter_endpoint)
elif alt_pkl_file_path == 2:
    trading_client = TradingClient(config.alpaca_paper_mega_api_key, config.alpaca_paper_mega_secret_key, paper=True)
    stock_client = StockHistoricalDataClient(config.alpaca_paper_mega_api_key, config.alpaca_paper_mega_secret_key)
    api = REST(config.alpaca_paper_mega_api_key, config.alpaca_paper_mega_secret_key, base_url=config.alpaca_paper_mega_endpoint)

account = trading_client.get_account()
print(account)

# Trading parameters
time_sensev = 0.50
time_incriment = 1
time_incriment5 = 10
symbol = tickers_L  # Your ticker list

print('time_incriment: ', time_incriment)
print('count: ', count)

# Market calendar setup
calendar = api.get_calendar()
now = datetime.datetime.utcnow()
today = datetime.datetime.today().date()

# Find the last trading day
last_trading_day = None
for day in reversed(calendar):
    if pd.Timestamp(day.date).date() < today:
        last_trading_day = pd.Timestamp(day.date, hour=12, tz='UTC')
        break

# Current time
now = datetime.datetime.now(pytz.UTC)
hours_since_last_noon = int((now - last_trading_day).total_seconds() / 3600) - 16

print(last_trading_day)
print(hours_since_last_noon)

# Lookback time logic
if str(datetime.datetime.utcnow().time()) < '13:29:59.000000':
    lookbak_time = 24
elif '08:00:00.971400' < str(datetime.datetime.utcnow().time()) < '16:00:00.971400' and str(datetime.datetime.utcnow().time()) < '20:00:00.971400':
    lookbak_time = 12
elif '16:00:00.971400' < str(datetime.datetime.utcnow().time()) and str(datetime.datetime.utcnow().time()) < '20:00:00.971400':
    lookbak_time = 16
elif str(datetime.datetime.utcnow().time()) > '20:00:00.971400':
    lookbak_time = 20
else:
    pass

print('lookbak_time', lookbak_time)
look_back_days = 0
look_back_days_vertiacl = 0

# Data collection with market hours filter
all_df_open = []
for i in range(1, 3, 1):
    print(i)
    look_back_days = i * 100
    print(f"look_back_days: {look_back_days}")
    print(f"\n")
    Histoic_start = datetime.datetime.today() - timedelta(days=look_back_days)
    print(f"Histoic_start: {Histoic_start}")
    if i > 0:
        end_time = datetime.datetime.today() - timedelta(days=look_back_days - 100)
    else:
        end_time = datetime.datetime.today() - timedelta(days=0)
    print(f"end_time: {end_time}")
    print(f"\n")

    # Get historical data
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        start=Histoic_start, 
        end=end_time, 
        timeframe=TimeFrame(time_incriment, TimeFrameUnit.Minute)
    )
    historic = stock_client.get_stock_bars(request_params)
    historic = historic.df
    historic = historic.bfill()
    historic.reset_index(inplace=True)
    historic['timestamp'] = pd.to_datetime(historic['timestamp'], unit='ms', utc=True)
    historic = historic.set_index('timestamp')

    historic = historic.drop_duplicates(keep='last')
    historic = historic.pivot(columns='symbol', values=historic.columns)

    historic.tail(50)

    # ✅ CRITICAL: Filter for market open hours (13:30 to 20:00)
    is_open = (historic.index.time >= datetime.datetime.strptime("13:30", "%H:%M").time()) & \
              (historic.index.time <= datetime.datetime.strptime("20:00", "%H:%M").time())
    df_open = historic[is_open]

    all_df_open.append(df_open)

# Combine all data
df_multiday = pd.concat(all_df_open)
df_multiday.sort_index(inplace=True)

# ✅ FIXED: Restructure to match cleaned_preprocessed.csv format
def restructure_to_environment_format(df_multiday):
    """
    Restructure the multi-index data to match the environment's expected format
    where coin is a column and date is a column, similar to cleaned_preprocessed.csv
    """
    print(" Restructuring data to environment format...")
    
    # Reset index to get timestamp as a column
    df_restructured = df_multiday.reset_index()
    df_restructured.rename(columns={'timestamp': 'date'}, inplace=True)
    
    # ✅ FIX: Ensure date column is properly formatted
    df_restructured['date'] = pd.to_datetime(df_restructured['date'])
    
    # Get all the symbol-specific columns
    symbol_cols = []
    for col in df_restructured.columns:
        if col != 'date' and isinstance(col, tuple) and len(col) == 2:
            symbol_cols.append(col)
    
    # Create the melted dataframe
    melted_data = []
    
    for idx, row in df_restructured.iterrows():
        # ✅ CRITICAL FIX: Extract actual datetime value, not Series
        date_val = row['date']
        if hasattr(date_val, 'iloc'):
            # If it's a Series, extract the first value
            date_val = date_val.iloc[0]
        elif hasattr(date_val, 'item'):
            # If it's a numpy scalar, convert to Python datetime
            date_val = date_val.item()
        
        # Get unique symbols from the column names
        symbols = set()
        for col in symbol_cols:
            if isinstance(col, tuple) and len(col) == 2:
                symbols.add(col[1])
        
        # Create a row for each symbol
        for symbol in symbols:
            row_data = {'date': date_val, 'coin': symbol}
            
            # Add all the features for this symbol
            for feature, sym in symbol_cols:
                if sym == symbol:
                    # ✅ FIX: Handle NaN values properly
                    value = row[(feature, sym)]
                    if pd.isna(value):
                        value = 0.0  # Default value for missing data
                    else:
                        # ✅ FIX: Extract scalar value if it's a Series
                        if hasattr(value, 'iloc'):
                            value = value.iloc[0]
                        elif hasattr(value, 'item'):
                            value = value.item()
                    row_data[feature] = value
            
            melted_data.append(row_data)
    
    # Convert to DataFrame
    df_final = pd.DataFrame(melted_data)
    
    # ✅ FIX: Ensure proper data types before sorting
    df_final['date'] = pd.to_datetime(df_final['date'])
    df_final['coin'] = df_final['coin'].astype(str)
    
    # Sort by date and coin
    df_final = df_final.sort_values(['date', 'coin']).reset_index(drop=True)
    
    # ✅ ADD: Create scaled features like in cleaned_preprocessed.csv
    print("📊 Creating scaled features...")
    
    # Define the base columns (raw OHLCV data)
    base_features = ['high', 'low', 'open', 'close', 'volume', 'quoteVolume', 'weightedAverage']
    
    # For each coin, create scaled features
    for coin in df_final['coin'].unique():
        coin_data = df_final[df_final['coin'] == coin].copy()
        
        if len(coin_data) > 0:
            # Calculate scaled features
            for feature in base_features:
                if feature in coin_data.columns:
                    # Create scaled version (similar to your cleaned_preprocessed.csv)
                    coin_data[f'{feature}_s'] = coin_data[feature].pct_change().fillna(0)
                    
                    # Create rolling averages
                    for window in [7, 14, 30]:
                        if len(coin_data) >= window:
                            coin_data[f'{feature}_s_roll_{window}'] = coin_data[f'{feature}_s'].rolling(window=window, min_periods=1).mean()
                        else:
                            coin_data[f'{feature}_s_roll_{window}'] = coin_data[f'{feature}_s']
            
            # Update the main dataframe
            df_final.loc[df_final['coin'] == coin, coin_data.columns] = coin_data
    
    # Fill any remaining NaN values
    df_final = df_final.fillna(0)
    
    print(f"✅ Restructured data shape: {df_final.shape}")
    print(f"📅 Date range: {df_final['date'].min()} to {df_final['date'].max()}")
    print(f" Coins: {df_final['coin'].unique()}")
    print(f" Sample data:")
    print(df_final.head(10))
    
    return df_final

# Restructure the data
df_environment_format = restructure_to_environment_format(df_multiday)

# Save the restructured data
df_environment_format.to_csv("cleaned_preprocessed_restructured.csv", index=False)
print(" Saved restructured data to 'cleaned_preprocessed_restructured.csv'")

# Verify the structure matches cleaned_preprocessed.csv
print("\n🔍 Verifying structure compatibility...")
print("Expected columns from cleaned_preprocessed.csv:")
expected_cols = ['date', 'coin', 'high', 'low', 'open', 'close', 'volume', 'quoteVolume', 'weightedAverage']
print(expected_cols)

print("\nActual columns in restructured data:")
actual_cols = df_environment_format.columns.tolist()
print(actual_cols[:20])  # Show first 20 columns

# Check if all expected columns are present
missing_cols = [col for col in expected_cols if col not in actual_cols]
if missing_cols:
    print(f"⚠️ Missing columns: {missing_cols}")
else:
    print("✅ All expected columns present!")

# Show sample of the restructured data
print("\n📊 Sample of restructured data:")
print(df_environment_format[['date', 'coin', 'high', 'low', 'open', 'close']].head(15))

# Analyze trading day structure
print("\n📈 Analyzing trading day structure...")
df_environment_format['day'] = df_environment_format['date'].dt.date

# Count steps per day
daily_counts = df_environment_format.groupby('day').size()
print(f"📊 Trading days analyzed: {len(daily_counts)}")
print(f"📊 Average steps per day: {daily_counts.mean():.1f}")
print(f" Min steps per day: {daily_counts.min()}")
print(f" Max steps per day: {daily_counts.max()}")
print(f"📊 Median steps per day: {daily_counts.median():.1f}")

# Show distribution
plt.figure(figsize=(12, 6))
plt.hist(daily_counts.values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(daily_counts.mean(), color='red', linestyle='--', label=f'Mean: {daily_counts.mean():.1f}')
plt.axvline(daily_counts.median(), color='orange', linestyle='--', label=f'Median: {daily_counts.median():.1f}')
plt.xlabel('Steps per Trading Day')
plt.ylabel('Frequency')
plt.title('Distribution of Steps per Trading Day (Restructured Data)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"\n🎯 Recommended episode length: {daily_counts.median():.0f} steps")
print(f"📊 Data ready for RL environment training!")

# Additional analysis for RL training
print("\n" + "="*60)
print("🧠 RL TRAINING ANALYSIS")
print("="*60)

# Calculate what percentage of days meet different thresholds
for threshold in [300, 350, 380, 390]:
    pct_above = (daily_counts >= threshold).mean() * 100
    print(f"📊 {pct_above:.1f}% of days have ≥{threshold} steps")

print(f"\n📈 DATA UTILIZATION:")
total_trading_days = len(daily_counts)
print(f"   • Total trading days: {total_trading_days}")
print(f"   • With random starts: ~{total_trading_days * 3} possible episodes")
print(f"   • With sliding windows: ~{total_trading_days * 5} possible episodes")

print(f"\n🎯 RECOMMENDED SETTINGS:")
print(f"   • Episode length: {daily_counts.median():.0f} steps")
print(f"   • Update config.py: FILE = 'cleaned_preprocessed_restructured.csv'")
print(f"   • Update config.py: EPISODE_LENGTH = {daily_counts.median():.0f}")

# %%
