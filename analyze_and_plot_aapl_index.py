import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Load the data
print("Loading data...")
df_multiday = pd.read_csv('cleaned_preprocessed_restructured.csv')
print(f"Data shape: {df_multiday.shape}")

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("\n" + "="*50)
print("DATA STRUCTURE ANALYSIS")
print("="*50)

# 1. Basic info about the dataset
print("\n1. BASIC DATASET INFO:")
print(f"Total rows: {len(df_multiday)}")
print(f"Total columns: {len(df_multiday.columns)}")
print(f"Memory usage: {df_multiday.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# 2. Column information
print("\n2. COLUMN NAMES:")
for i, col in enumerate(df_multiday.columns):
    print(f"{i:2d}: {col}")

# 3. Data types
print("\n3. DATA TYPES:")
print(df_multiday.dtypes)

# 4. Check for missing values
print("\n4. MISSING VALUES:")
missing_values = df_multiday.isnull().sum()
print(missing_values[missing_values > 0])

# 5. Unique coins and their counts
print("\n5. UNIQUE COINS AND COUNTS:")
coin_counts = df_multiday['coin'].value_counts()
print(f"Total unique coins: {len(coin_counts)}")
print("\nTop 10 coins by frequency:")
print(coin_counts.head(10))

# 6. Time range analysis
print("\n6. TIME RANGE ANALYSIS:")
df_multiday['date'] = pd.to_datetime(df_multiday['date'])
print(f"Date range: {df_multiday['date'].min()} to {df_multiday['date'].max()}")
print(f"Total unique timestamps: {df_multiday['date'].nunique()}")

# 7. AAPL specific analysis
print("\n7. AAPL SPECIFIC ANALYSIS:")
aapl_data = df_multiday[df_multiday['coin'] == 'AAPL'].copy()
print(f"AAPL data shape: {aapl_data.shape}")
print(f"AAPL unique timestamps: {aapl_data['date'].nunique()}")

if len(aapl_data) > 0:
    print(f"AAPL date range: {aapl_data['date'].min()} to {aapl_data['date'].max()}")
    print(f"AAPL price range - High: {aapl_data['high'].min():.2f} to {aapl_data['high'].max():.2f}")
    print(f"AAPL price range - Low: {aapl_data['low'].min():.2f} to {aapl_data['low'].max():.2f}")
    print(f"AAPL price range - Close: {aapl_data['close'].min():.2f} to {aapl_data['close'].max():.2f}")
    
    # Check if we have time series data or just snapshots
    aapl_data_sorted = aapl_data.sort_values('date')
    print(f"\nFirst 5 AAPL records:")
    print(aapl_data_sorted[['date', 'coin', 'open', 'high', 'low', 'close', 'volume']].head())
    
    print(f"\nLast 5 AAPL records:")
    print(aapl_data_sorted[['date', 'coin', 'open', 'high', 'low', 'close', 'volume']].tail())

# 8. Sample data structure
print("\n8. SAMPLE DATA STRUCTURE:")
print("First 3 rows of the dataset:")
print(df_multiday.head(3))

# 9. Check for time series structure
print("\n9. TIME SERIES STRUCTURE ANALYSIS:")
# Group by date and count coins per timestamp
coins_per_timestamp = df_multiday.groupby('date')['coin'].count()
print(f"Coins per timestamp - Min: {coins_per_timestamp.min()}, Max: {coins_per_timestamp.max()}")
print(f"Average coins per timestamp: {coins_per_timestamp.mean():.2f}")

# Check if all timestamps have the same number of coins
if coins_per_timestamp.nunique() == 1:
    print("✓ All timestamps have the same number of coins - good for time series analysis")
else:
    print("⚠ Timestamps have varying numbers of coins - may need special handling")

print("\n" + "="*50)
print("PLOTTING AAPL DATA WITH INDEX-BASED X-AXIS")
print("="*50)

# Plot AAPL data using index instead of date
if len(aapl_data) > 0:
    # Sort by date and reset index to get sequential numbering
    aapl_data_sorted = aapl_data.sort_values('date').reset_index(drop=True)
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Price data
    plt.subplot(2, 2, 1)
    plt.plot(aapl_data_sorted.index, aapl_data_sorted['close'], label='Close', linewidth=2)
    plt.plot(aapl_data_sorted.index, aapl_data_sorted['high'], label='High', alpha=0.7)
    plt.plot(aapl_data_sorted.index, aapl_data_sorted['low'], label='Low', alpha=0.7)
    plt.plot(aapl_data_sorted.index, aapl_data_sorted['open'], label='Open', alpha=0.7)
    plt.title('AAPL Price Data Over Time (Index-based)')
    plt.xlabel('Time Step Index')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Volume
    plt.subplot(2, 2, 2)
    plt.plot(aapl_data_sorted.index, aapl_data_sorted['volume'], color='orange', linewidth=2)
    plt.title('AAPL Trading Volume Over Time (Index-based)')
    plt.xlabel('Time Step Index')
    plt.ylabel('Volume')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Price range (High-Low)
    plt.subplot(2, 2, 3)
    price_range = aapl_data_sorted['high'] - aapl_data_sorted['low']
    plt.plot(aapl_data_sorted.index, price_range, color='red', linewidth=2)
    plt.title('AAPL Daily Price Range (High - Low)')
    plt.xlabel('Time Step Index')
    plt.ylabel('Price Range ($)')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Close price with moving averages
    plt.subplot(2, 2, 4)
    plt.plot(aapl_data_sorted.index, aapl_data_sorted['close'], label='Close Price', linewidth=2)
    
    # Add moving averages if we have enough data
    if len(aapl_data_sorted) > 7:
        ma_7 = aapl_data_sorted['close'].rolling(window=7).mean()
        plt.plot(aapl_data_sorted.index, ma_7, label='7-period MA', alpha=0.7)
    
    if len(aapl_data_sorted) > 14:
        ma_14 = aapl_data_sorted['close'].rolling(window=14).mean()
        plt.plot(aapl_data_sorted.index, ma_14, label='14-period MA', alpha=0.7)
    
    if len(aapl_data_sorted) > 30:
        ma_30 = aapl_data_sorted['close'].rolling(window=30).mean()
        plt.plot(aapl_data_sorted.index, ma_30, label='30-period MA', alpha=0.7)
    
    plt.title('AAPL Close Price with Moving Averages (Index-based)')
    plt.xlabel('Time Step Index')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('aapl_analysis_plot_index_based.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ AAPL plots created and saved as 'aapl_analysis_plot_index_based.png'")
    
    # Additional insights
    print(f"\nAAPL TRADING INSIGHTS:")
    print(f"Total trading sessions: {len(aapl_data_sorted)}")
    print(f"Average daily volume: {aapl_data_sorted['volume'].mean():,.0f}")
    print(f"Average daily price range: ${price_range.mean():.2f}")
    print(f"Price volatility (std of close): ${aapl_data_sorted['close'].std():.2f}")
    
    # Show some key statistics
    print(f"\nAAPL PRICE STATISTICS:")
    print(f"Starting price: ${aapl_data_sorted['close'].iloc[0]:.2f}")
    print(f"Ending price: ${aapl_data_sorted['close'].iloc[-1]:.2f}")
    print(f"Total return: {((aapl_data_sorted['close'].iloc[-1] / aapl_data_sorted['close'].iloc[0]) - 1) * 100:.2f}%")
    print(f"Maximum price: ${aapl_data_sorted['high'].max():.2f}")
    print(f"Minimum price: ${aapl_data_sorted['low'].min():.2f}")
    
else:
    print("❌ No AAPL data found in the dataset")

print("\n" + "="*50)
print("DATA STRUCTURE SUMMARY")
print("="*50)
print("Based on the analysis:")
print(f"- Dataset contains {len(coin_counts)} different coins")
print(f"- Time range: {df_multiday['date'].min()} to {df_multiday['date'].max()}")
print(f"- Each timestamp contains {coins_per_timestamp.iloc[0]} coins")
print(f"- Total timestamps: {df_multiday['date'].nunique()}")
print(f"- AAPL records: {len(aapl_data)}")
print(f"- Using index-based plotting to avoid gaps between trading days")
