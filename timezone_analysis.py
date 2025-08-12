import pandas as pd
from datetime import datetime
import pytz

# Load the data
print("Loading data...")
df_multiday = pd.read_csv('cleaned_preprocessed_restructured.csv')
print(f"Data shape: {df_multiday.shape}")

# Convert date column to datetime
df_multiday['date'] = pd.to_datetime(df_multiday['date'])

print("\n" + "="*50)
print("TIMEZONE ANALYSIS")
print("="*50)

# 1. Examine the date format
print("\n1. DATE FORMAT EXAMINATION:")
print("First 5 dates:")
print(df_multiday['date'].head())

print("\nLast 5 dates:")
print(df_multiday['date'].tail())

# 2. Check if timezone info is present
print("\n2. TIMEZONE INFORMATION:")
print(f"Timezone info present: {df_multiday['date'].dt.tz is not None}")
if df_multiday['date'].dt.tz is not None:
    print(f"Timezone: {df_multiday['date'].dt.tz}")
else:
    print("No timezone info - dates are timezone-naive")

# 3. Extract time components
print("\n3. TIME COMPONENTS ANALYSIS:")
df_multiday['hour'] = df_multiday['date'].dt.hour
df_multiday['minute'] = df_multiday['date'].dt.minute
df_multiday['day_of_week'] = df_multiday['date'].dt.day_name()

print("Hour distribution:")
hour_counts = df_multiday['hour'].value_counts().sort_index()
print(hour_counts)

print("\nMinute distribution (first 10):")
minute_counts = df_multiday['minute'].value_counts().sort_index().head(10)
print(minute_counts)

print("\nDay of week distribution:")
day_counts = df_multiday['day_of_week'].value_counts()
print(day_counts)

# 4. Check for market hours patterns
print("\n4. MARKET HOURS ANALYSIS:")
print("Looking for patterns that indicate market hours...")

# Check if we have data in the 9:30-16:00 range (EST market hours)
est_market_start = 9
est_market_end = 16
utc_market_start = 14  # 9:30 EST = 14:30 UTC (assuming EST is UTC-5)
utc_market_end = 21    # 16:00 EST = 21:00 UTC

print(f"\nData in EST market hours (9-16): {len(df_multiday[df_multiday['hour'].between(est_market_start, est_market_end)])}")
print(f"Data in UTC market hours (14-21): {len(df_multiday[df_multiday['hour'].between(utc_market_start, utc_market_end)])}")

# Check the actual hour range in the data
print(f"Actual hour range in data: {df_multiday['hour'].min()} to {df_multiday['hour'].max()}")

# 5. Sample data by hour to see patterns
print("\n5. SAMPLE DATA BY HOUR:")
for hour in sorted(df_multiday['hour'].unique()):
    sample_data = df_multiday[df_multiday['hour'] == hour].head(3)
    print(f"\nHour {hour}:00 - Sample dates:")
    for _, row in sample_data.iterrows():
        print(f"  {row['date']} - {row['coin']}")

# 6. Determine timezone based on patterns
print("\n6. TIMEZONE DETERMINATION:")
if df_multiday['hour'].min() >= 13 and df_multiday['hour'].max() <= 21:
    print("✓ Data appears to be in UTC (market hours 13:30-21:00)")
    print("  EST market hours (9:30-16:00) would be 14:30-21:00 in UTC")
    timezone = "UTC"
elif df_multiday['hour'].min() >= 9 and df_multiday['hour'].max() <= 16:
    print("✓ Data appears to be in EST (market hours 9:30-16:00)")
    timezone = "EST"
else:
    print("⚠ Cannot clearly determine timezone from hour patterns")
    print(f"  Hour range: {df_multiday['hour'].min()}:00 to {df_multiday['hour'].max()}:00")
    timezone = "Unknown"

print(f"\nCONCLUSION: Data appears to be in {timezone} timezone")

# 7. Show market hours in both timezones for reference
print("\n7. MARKET HOURS REFERENCE:")
print("EST Market Hours: 9:30 AM - 4:00 PM EST")
print("UTC Market Hours: 2:30 PM - 9:00 PM UTC (when EST is UTC-5)")
print("UTC Market Hours: 1:30 PM - 8:00 PM UTC (when EST is UTC-4, during DST)")

# 8. Check for specific market open/close times
print("\n8. MARKET OPEN/CLOSE TIME ANALYSIS:")
market_open_est = 9.5  # 9:30 AM
market_close_est = 16.0  # 4:00 PM

if timezone == "UTC":
    market_open_utc = 14.5  # 2:30 PM UTC
    market_close_utc = 21.0  # 9:00 PM UTC
    
    # Check for data around these times
    open_data = df_multiday[df_multiday['hour'] == 14]
    close_data = df_multiday[df_multiday['hour'] == 21]
    
    print(f"Data at UTC 14:00 (market open): {len(open_data)} records")
    print(f"Data at UTC 21:00 (market close): {len(close_data)} records")
    
elif timezone == "EST":
    # Check for data around these times
    open_data = df_multiday[df_multiday['hour'] == 9]
    close_data = df_multiday[df_multiday['hour'] == 16]
    
    print(f"Data at EST 9:00 (market open): {len(open_data)} records")
    print(f"Data at EST 16:00 (market close): {len(close_data)} records")

print("\n" + "="*50)
print("FINAL RECOMMENDATION")
print("="*50)
print(f"Based on the analysis, your data appears to be in {timezone} timezone.")
print("For plotting vertical lines at market open/close:")
if timezone == "UTC":
    print("- Market Open: 14:30 UTC (9:30 AM EST)")
    print("- Market Close: 21:00 UTC (4:00 PM EST)")
elif timezone == "EST":
    print("- Market Open: 9:30 EST")
    print("- Market Close: 16:00 EST")
else:
    print("- Need to manually verify timezone based on your data source")
