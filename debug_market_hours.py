import pandas as pd
from datetime import datetime
import pytz

# Load the data
print("Loading data...")
df_multiday = pd.read_csv('cleaned_preprocessed_restructured.csv')
print(f"Data shape: {df_multiday.shape}")

# Convert date column to datetime
df_multiday['date'] = pd.to_datetime(df_multiday['date'])

print("\n" + "="*60)
print("MARKET HOURS DEBUGGING")
print("="*60)

# 1. Examine the actual data time range
print("\n1. ACTUAL DATA TIME RANGE:")
print(f"Data starts at: {df_multiday['date'].min()}")
print(f"Data ends at: {df_multiday['date'].max()}")

# 2. Extract time components
df_multiday['hour'] = df_multiday['date'].dt.hour
df_multiday['minute'] = df_multiday['date'].dt.minute
df_multiday['time_decimal'] = df_multiday['hour'] + df_multiday['minute'] / 60.0

print(f"\n2. TIME COMPONENTS ANALYSIS:")
print(f"Hour range: {df_multiday['hour'].min()}:00 to {df_multiday['hour'].max()}:00")
print(f"Minute range: {df_multiday['minute'].min()}:00 to {df_multiday['minute'].max()}:00")

# 3. Check the actual time distribution
print(f"\n3. TIME DISTRIBUTION:")
hour_counts = df_multiday['hour'].value_counts().sort_index()
print("Hour distribution:")
for hour, count in hour_counts.items():
    print(f"  {hour:2d}:00 - {count:6d} records")

# 4. Analyze the issue
print(f"\n4. THE PROBLEM ANALYSIS:")
print("Based on the data creation code in paca_data.ipynb:")
print("  - Data was filtered from 13:30 to 20:00 UTC")
print("  - This means the data starts at 13:30 UTC, not 14:30 UTC")
print("  - The data ends at 20:00 UTC, not 21:00 UTC")

print(f"\n5. CORRECT MARKET HOURS:")
print("EST Market Hours: 9:30 AM - 4:00 PM EST")
print("UTC Market Hours (EST-5): 14:30 - 21:00 UTC")
print("UTC Market Hours (EST-4, DST): 13:30 - 20:00 UTC")

print(f"\n6. DATA VS EXPECTED MARKET HOURS:")
print("Data range: 13:30-20:00 UTC")
print("Expected EST-5: 14:30-21:00 UTC")
print("Expected EST-4: 13:30-20:00 UTC")

# 7. Determine the correct market hours
print(f"\n7. CONCLUSION:")
if df_multiday['hour'].min() == 13 and df_multiday['hour'].max() == 20:
    print("✓ Data matches EST-4 (DST) market hours: 13:30-20:00 UTC")
    print("  This means the data is using Daylight Saving Time (EST-4)")
    correct_market_open = 13.5  # 13:30 UTC
    correct_market_close = 20.0  # 20:00 UTC
    timezone_info = "EST-4 (DST)"
elif df_multiday['hour'].min() == 14 and df_multiday['hour'].max() == 21:
    print("✓ Data matches EST-5 market hours: 14:30-21:00 UTC")
    print("  This means the data is using Standard Time (EST-5)")
    correct_market_open = 14.5  # 14:30 UTC
    correct_market_close = 21.0  # 21:00 UTC
    timezone_info = "EST-5 (Standard)"
else:
    print("⚠ Data doesn't match expected market hours")
    print(f"  Actual range: {df_multiday['hour'].min()}:30 to {df_multiday['hour'].max()}:00")
    correct_market_open = df_multiday['hour'].min() + 0.5
    correct_market_close = df_multiday['hour'].max()
    timezone_info = "Unknown"

print(f"\n8. CORRECTED MARKET LINE TIMES:")
print(f"Market Open: {correct_market_open:.1f} UTC ({int(correct_market_open)}:{int((correct_market_open % 1) * 60):02d} UTC)")
print(f"Market Close: {correct_market_close:.1f} UTC ({int(correct_market_close)}:{int((correct_market_close % 1) * 60):02d} UTC)")

# 9. Test the corrected logic
print(f"\n9. TESTING CORRECTED LOGIC:")
aapl_data = df_multiday[df_multiday['coin'] == 'AAPL'].copy()
aapl_data_sorted = aapl_data.sort_values('date').reset_index(drop=True)

# Add time components
aapl_data_sorted['hour'] = aapl_data_sorted['date'].dt.hour
aapl_data_sorted['minute'] = aapl_data_sorted['date'].dt.minute
aapl_data_sorted['time_decimal'] = aapl_data_sorted['hour'] + aapl_data_sorted['minute'] / 60.0

# Find market open and close indices with corrected times
market_open_indices = []
market_close_indices = []

for date, day_data in aapl_data_sorted.groupby(aapl_data_sorted['date'].dt.date):
    # Find market open (first occurrence at or after correct time)
    open_mask = day_data['time_decimal'] >= correct_market_open
    if open_mask.any():
        market_open_idx = day_data[open_mask].index[0]
        market_open_indices.append(market_open_idx)
        print(f"  {date}: Market open at index {market_open_idx} (time: {day_data.loc[market_open_idx, 'date']})")
    
    # Find market close (last occurrence before or at correct time)
    close_mask = day_data['time_decimal'] <= correct_market_close
    if close_mask.any():
        market_close_idx = day_data[close_mask].index[-1]
        market_close_indices.append(market_close_idx)
        print(f"  {date}: Market close at index {market_close_idx} (time: {day_data.loc[market_close_idx, 'date']})")

print(f"\n10. SUMMARY:")
print(f"Data timezone: {timezone_info}")
print(f"Correct market open: {correct_market_open:.1f} UTC")
print(f"Correct market close: {correct_market_close:.1f} UTC")
print(f"Found {len(market_open_indices)} market open lines")
print(f"Found {len(market_close_indices)} market close lines")

print(f"\n11. WHY THE VOLATILITY SPIKES APPEARED ON RED LINES:")
print("The issue was that I was marking market open at 14:30 UTC when the data")
print("actually starts at 13:30 UTC. This caused the market open lines (green)")
print("to appear in the middle of the trading day, making the early volatility")
print("spikes appear to be 'after market open' when they were actually at the")
print("true market open time.")
