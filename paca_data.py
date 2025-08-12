tickers = ['''
MMM,
AOS,
ABT,
ABBV,
ACN,
ADBE,
AMD,
AES,
AFL,
A,
APD,
ABNB,
AKAM,
ALB,
ARE,
ALGN,
ALLE,
LNT,
ALL,
GOOGL,
GOOG,
MO,
AMZN,
AMCR,
AMTM,
AEE,
AEP,
AXP,
AIG,
AMT,
AWK,
AMP,
AME,
AMGN,
APH,
ADI,
ANSS,
AON,
APA,
AAPL,
AMAT,
APTV,
ACGL,
ADM,
ANET,
AJG,
AIZ,
T,
ATO,
ADSK,
ADP,
AZO,
AVB,
AVY,
AXON,
BKR,
BALL,
BAC,
BAX,
BDX,
BBY,
TECH,
BIIB,
BLK,
BX,
BK,
BA,
BKNG,
BWA,
BSX,
BMY,
AVGO,
BR,
BRO,
BLDR,
BG,
BXP,
CHRW,
CDNS,
CZR,
CPT,
CPB,
COF,
CRWV,
CAH,
KMX,
CCL,
CARR,
CTLT,
CAT,
CBOE,
CBRE,
CDW,
CE,
COR,
CNC,
CNP,
CF,
CRL,
SCHW,
CHTR,
CVX,
CMG,
CB,
CHD,
CI,
CINF,
CTAS,
CSCO,
C,
CFG,
CLX,
CME,
CMS,
KO,
CTSH,
CL,
CMCSA,
CAG,
COP,
ED,
STZ,
CEG,
COO,
CPRT,
GLW,
CPAY,
CTVA,
CSGP,
COST,
CTRA,
CRWD,
CCI,
CSX,
CMI,
CVS,
DHR,
DRI,
DVA,
DAY,
DECK,
DE,
DELL,
DAL,
DVN,
DXCM,
FANG,
DLR,
DFS,
DG,
DLTR,
D,
DPZ,
DOV,
DOW,
DHI,
DTE,
DUK,
DD,
EMN,
ETN,
EBAY,
ECL,
EIX,
EW,
EA,
ELV,
EMR,
ENPH,
ETR,
EOG,
EPAM,
EQT,
EFX,
EQIX,
EQR,
ERIE,
ESS,
EL,
EG,
EVRG,
ES,
EXC,
EXPE,
EXPD,
EXR,
XOM,
FFIV,
FDS,
FICO,
FAST,
FRT,
FDX,
FIS,
FITB,
FSLR,
FE,
FI,
FMC,
F,
FTNT,
FTV,
FOXA,
FOX,
BEN,
FCX,
GRMN,
IT,
GE,
GEHC,
GEV,
GEN,
GNRC,
GD,
GIS,
GM,
GPC,
GILD,
GPN,
GL,
GDDY,
GS,
HAL,
HIG,
HAS,
HCA,
DOC,
HSIC,
HSY,
HES,
HPE,
HLT,
HOLX,
HD,
HON,
HRL,
HST,
HWM,
HPQ,
HUBB,
HUM,
HBAN,
HII,
IBM,
IEX,
IDXX,
ITW,
INCY,
IR,
PODD,
INTC,
ICE,
IFF,
IP,
IPG,
INTU,
ISRG,
IVZ,
INVH,
IQV,
IRM,
JBHT,
JBL,
JKHY,
J,
JNJ,
JCI,
JPM,
JNPR,
K,
KVUE,
KDP,
KEY,
KEYS,
KMB,
KIM,
KMI,
KKR,
KLAC,
KHC,
KR,
LHX,
LH,
LRCX,
LW,
LVS,
LDOS,
LEN,
LLY,
LIN,
LYV,
LKQ,
LMT,
L,
LOW,
LULU,
LYB,
MTB,
MRO,
MPC,
MKTX,
MAR,
MMC,
MLM,
MAS,
MA,
MTCH,
MKC,
MCD,
MCK,
MDT,
MRK,
META,
MET,
MTD,
MGM,
MCHP,
MU,
MSFT,
MAA,
MRNA,
MHK,
MOH,
TAP,
MDLZ,
MPWR,
MNST,
MCO,
MS,
MOS,
MSI,
MSCI,
NDAQ,
NTAP,
NFLX,
NEM,
NWSA,
NWS,
NEE,
NKE,
NI,
NDSN,
NSC,
NTRS,
NOC,
NCLH,
NRG,
NUE,
NVDA,
NVR,
NXPI,
ORLY,
OXY,
ODFL,
OMC,
ON,
OKE,
ORCL,
OTIS,
PCAR,
PKG,
PLTR,
PANW,
PARA,
PH,
PAYX,
PAYC,
PYPL,
PNR,
PEP,
PFE,
PCG,
PM,
PSX,
PNW,
PNC,
POOL,
PPG,
PPL,
PFG,
PG,
PGR,
PLD,
PRU,
PEG,
PTC,
PSA,
PHM,
QRVO,
PWR,
QCOM,
DGX,
RL,
RJF,
RTX,
O,
REG,
REGN,
RF,
RSG,
RMD,
RVTY,
ROK,
ROL,
ROP,
ROST,
RCL,
SPGI,
CRM,
SBAC,
SLB,
STX,
SRE,
NOW,
SHW,
SPG,
SWKS,
SJM,
SW,
SNA,
SOLV,
SO,
LUV,
SWK,
SBUX,
STT,
STLD,
STE,
SYK,
SMCI,
SYF,
SNPS,
SYY,
TMUS,
TROW,
TTWO,
TPR,
TRGP,
TGT,
TEL,
TDY,
TFX,
TER,
TSLA,
TXN,
TXT,
TMO,
TJX,
TSCO,
TT,
TDG,
TRV,
TRMB,
TFC,
TYL,
TSN,
USB,
UBER,
UDR,
ULTA,
UNP,
UAL,
UPS,
URI,
UNH,
UHS,
VLO,
VTR,
VLTO,
VRSN,
VRSK,
VZ,
VRTX,
VTRS,
VICI,
V,
VST,
VMC,
WRB,
GWW,
WAB,
WBA,
WMT,
DIS,
WBD,
WM,
WAT,
WEC,
WFC,
WELL,
WST,
WDC,
WY,
XYL,
YUM,
ZBRA,
ZBH,
ZTS,    
''']
import pandas as pd
import numpy as np

# Read the file
with open("embedding_nyse.csv", 'r') as file:
    lines = file.readlines()

# Extract header
header = lines[0].strip().split(',')

# Parse data
data = []
for line in lines[1:]:
    parts = line.strip().split('\t')
    if len(parts) >= len(header):
        row = dict(zip(header, parts[:len(header)]))
        data.append(row)

# Create DataFrame
nyse_list = pd.DataFrame(data)

# Function to convert market cap and volume to float
def convert_market_cap(value):
    if value == '-':
        return np.nan
    value = value.replace(',', '')
    if value.endswith('B'):
        return float(value[:-1]) * 1e9
    elif value.endswith('M'):
        return float(value[:-1]) * 1e6
    elif value.endswith('K'):
        return float(value[:-1]) * 1e3
    else:
        return float(value)

# Function to convert change to float
def convert_change(value):
    if value == '-':
        return np.nan
    return float(value.rstrip('%')) / 100

# Convert columns to appropriate types
nyse_list['no'] = pd.to_numeric(nyse_list['no'], errors='coerce')
nyse_list['market_cap'] = nyse_list['market_cap'].apply(convert_market_cap)
nyse_list['price'] = pd.to_numeric(nyse_list['price'].str.replace(',', ''), errors='coerce')
nyse_list['change'] = nyse_list['change'].apply(convert_change)
nyse_list['volume'] = nyse_list['volume'].apply(convert_market_cap)

# Display column names and data types
print("\nColumns:")
print(nyse_list.dtypes)

# Display first few rows
print("\nFirst few rows:")
print(nyse_list.head().to_string())


nyse_list.tail()
# print(nyse_list)
# print(nyse_list.market_cap.min())
# Get the top 10 companies by market cap
top_10 = nyse_list.nlargest(10, 'market_cap')

# Print the symbols and market caps
print("Top 10 companies by market cap:")
for index, row in top_10.iterrows():
    print(f"{row['symbol']}: ${row['market_cap']:,.0f}")

# Calculate and print the range
range_market_cap = nyse_list.market_cap.max() - nyse_list.market_cap.min()
print(f"\nMarket Cap Range: ${range_market_cap:,.0f}")
range_market_cap = nyse_list.market_cap.max() - nyse_list.market_cap.min()
print(range_market_cap)

tickers = [f"{tic.strip()}" for tic in tickers[0].split(',') if tic.strip()]

# Print original tickers

print(len(tickers))
print(f'tickers = {tickers}')

# Extract symbols from nyse_list and combine with original tickers
nyse_symbols = nyse_list['symbol'].tolist()
combined_tickers = list(set(tickers + nyse_symbols))

# Print combined tickers
print(len(combined_tickers))
print(f'combined_tickers = {combined_tickers}')

# Initial list
tickers_L = [] #['AAPL', 'GOOGL', 'MSFT', 'SPY', 'QQQ']

# Additional symbols provided
additional_symbols = [
    'TQQQ', 'SQQQ', 'NVDA', 'USO', 'TSLA', 'SCO', 'UNH', 'YINN', 'YANG',
    'NUGT', 'DUST', 'UVXY', 'KOLD', 'BOIL', 'LABU', 'LABD', 'TNA', 'TZA',
    'FAS', 'FAZ', 'SOXL', 'SOXS', 'ERY', 'ERX'
]

# Well-known large market cap stocks from each sector
large_market_cap_stocks = {
    'Technology': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META'],
    'Healthcare': ['UNH', 'JNJ', 'PFE', 'MRK', 'ABBV'],
    'Financials': ['JPM', 'BAC', 'WFC', 'C', 'GS'],
    'Consumer Discretionary': ['AMZN', 'TSLA', 'HD', 'NKE', 'MCD'],
    'Communication Services': ['GOOGL', 'META', 'NFLX', 'DIS', 'VZ'],
    'Industrials': ['HON', 'BA', 'GE', 'CAT', 'UPS'],
    'Consumer Staples': ['PG', 'KO', 'PEP', 'WMT', 'COST'],
    'Energy': ['XOM', 'CVX', 'COP', 'PSX', 'MPC'],
    'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP'],
    'Real Estate': ['PLD', 'AMT', 'CCI', 'EQIX', 'PSA'],
    'Materials': ['LIN', 'APD', 'ECL', 'SHW', 'NEM'],
}

# Well-known large market cap ETFs from each sector
large_market_cap_etfs = {
    'Technology': ['XLK', 'VGT'],
    'Healthcare': ['XLV', 'VHT'],
    'Financials': ['XLF', 'VFH'],
    'Consumer Discretionary': ['XLY', 'VCR'],
    'Communication Services': ['XLC', 'VOX'],
    'Industrials': ['XLI', 'VIS'],
    'Consumer Staples': ['XLP', 'VDC'],
    'Energy': ['XLE', 'VDE'],
    'Utilities': ['XLU', 'VPU'],
    'Real Estate': ['XLRE', 'VNQ'],
    'Materials': ['XLB', 'VAW'],
}

# Combine all tickers
tickers_L.extend(additional_symbols)
for sector_stocks in large_market_cap_stocks.values():
    tickers_L.extend(sector_stocks)
for sector_etfs in large_market_cap_etfs.values():
    tickers_L.extend(sector_etfs)

# Ensure no duplicates 
tickers_L = list(set(tickers_L))
print(len(tickers_L))
print("tickers_L =", tickers_L)
import os 
import pandas as pd
text_file = open("all_etf.txt", "r") # all_stockanalysis.txt 
etf_ticker = text_file.read()
text_file.close()
etf_ticker = etf_ticker.split("\n")
etf_ticker = [ticker.strip() for ticker in etf_ticker if ticker.strip()]

# Extract only the ticker symbols (first column before tab)
etf_symbols = []
for line in etf_ticker:
    if line.strip():  # Skip empty lines
        symbol = line.split("\t")[0].strip()  # Split by tab and take first part
        etf_symbols.append(symbol)
        print(symbol)

print(f"combined_tickers: {len(combined_tickers)}")
combined_tickers.extend(etf_symbols)
print(f"etf_symbols: {len(etf_symbols)}")
combined_tickers = list(set(combined_tickers+etf_symbols))
print(f"new combined_tickers: {len(combined_tickers)}")

import os 
import pandas as pd
text_file = open("all_stockanalysis.txt", "r") # all_stockanalysis.txt 
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
print(f"combined_tickers: {len(combined_tickers)}")
combined_tickers.extend(ticker_symbols)
print(f"ticker_symbols: {len(ticker_symbols)}")
combined_tickers = list(set(combined_tickers+ticker_symbols))
print(f"new combined_tickers: {len(combined_tickers)}")

read_file = open("all_tickers.txt", "r")
read_file = read_file.read()
read_file = read_file.split("\n")
read_file = [ticker.strip() for ticker in read_file if ticker.strip()]
potential_tickers = []
for symbol in read_file:
    print(symbol)
    potential_tickers.append(symbol)

print(f"\nTotal ETF symbols extracted: {len(potential_tickers)}")
print(f"combined_tickers: {len(combined_tickers)}")
combined_tickers.extend(potential_tickers)
print(f"ticker_symbols: {len(potential_tickers)}")
combined_tickers = list(set(combined_tickers+potential_tickers))
print(f"new combined_tickers: {len(combined_tickers)}")
combined_tickers.sort()
print(combined_tickers)
print(f"combined_tickers: {len(combined_tickers)}")

# Remove duplicates using Python's built-in methods
combined_tickers_unique = list(set(combined_tickers))
combined_tickers_unique.sort()  # Sort to maintain order

print(f"reduce duplicates: {combined_tickers_unique}")
print(f"reduce duplicates length: {len(combined_tickers_unique)}")

# Update the original list to be unique
combined_tickers = combined_tickers_unique




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
# Define market hours in EST (pytz handles DST automatically)
est_tz = pytz.timezone('US/Eastern')
utc_tz = pytz.UTC

# Market hours in EST
MARKET_OPEN_EST = datetime.time(9, 30)   # 9:30 AM EST
MARKET_CLOSE_EST = datetime.time(16, 0)  # 4:00 PM EST

def est_to_utc(est_time, date):
    """Convert EST time to UTC for a specific date (handles DST automatically)"""
    # Create a datetime in EST for the given date and time
    est_dt = est_tz.localize(datetime.datetime.combine(date, est_time))
    # Convert to UTC
    utc_dt = est_dt.astimezone(utc_tz)
    return utc_dt.time()

def get_market_hours_utc(date):
    """Get market open/close times in UTC for a specific date"""
    open_utc = est_to_utc(MARKET_OPEN_EST, date)
    close_utc = est_to_utc(MARKET_CLOSE_EST, date)
    return open_utc, close_utc

def get_market_datetime_utc(date):
    """Get market open/close datetime objects in UTC for a specific date (handles DST automatically)"""
    try:
        # Create a datetime in EST for the given date and time
        est_dt_open = est_tz.localize(datetime.datetime.combine(date, MARKET_OPEN_EST))
        est_dt_close = est_tz.localize(datetime.datetime.combine(date, MARKET_CLOSE_EST))
        
        # Convert to UTC
        utc_dt_open = est_dt_open.astimezone(utc_tz)
        utc_dt_close = est_dt_close.astimezone(utc_tz)
        
        return utc_dt_open, utc_dt_close
    except Exception as e:
        print(f"⚠️ Error converting timezone for {date}: {e}")
        # Fallback to hardcoded values (should not happen with proper DST handling)
        # utc_open = datetime.datetime.combine(date, datetime.time(13, 30)).replace(tzinfo=utc_tz)
        # utc_close = datetime.datetime.combine(date, datetime.time(20, 0)).replace(tzinfo=utc_tz)
        # return utc_open, utc_close


def get_recent_trading_days(calendar, n_days: int, offset_blocks: int = 0) -> list:
    """Return the last n_days trading dates, optionally offset by blocks of n_days.

    - calendar: Alpaca market calendar previously fetched
    - n_days: number of trading days to return per block
    - offset_blocks: 0 returns the most recent block, 1 returns the previous block of n_days, etc.
    """
    today = datetime.datetime.today().date()
    trading_days_all = [pd.Timestamp(day.date).date() for day in calendar if pd.Timestamp(day.date).date() < today]
    trading_days_all.sort()

    end_index = len(trading_days_all) - (offset_blocks * n_days)
    start_index = max(0, end_index - n_days)
    return trading_days_all[start_index:end_index]

look_back_days = 0
look_back_days_vertiacl = 0

# Limited to 100 so if past trading day 99 for getting 9:30 to 16:00 start end
# Data collection with market hours filter
# ✅ FIXED: Data collection with proper timezone handling for blocks of trading days
all_df_open = []
DAYS_PER_BLOCK = 6
for i in range(1, 3, 1):
    print(f"\n{'='*60}")
    print(f"Processing iteration {i} (block of {DAYS_PER_BLOCK} trading days)")
    print(f"{'='*60}")

    # Get the i-th most recent block of DAYS_PER_BLOCK trading days
    trading_days = get_recent_trading_days(calendar, n_days=DAYS_PER_BLOCK, offset_blocks=i-1)
    print(f"Trading days in block {i}: {trading_days}")

    for historical_date in trading_days:
        # ✅ FIXED: Get market hours for THIS SPECIFIC DATE (handles DST properly)
        utc_dt_open, utc_dt_close = get_market_datetime_utc(historical_date)

        print(f"Market hours for {historical_date}:")
        print(f"  EST: {MARKET_OPEN_EST} - {MARKET_CLOSE_EST}")
        print(f"  UTC: {utc_dt_open.time()} - {utc_dt_close.time()}")
        print(f"  UTC datetime: {utc_dt_open} - {utc_dt_close}")

        # ✅ Use proper UTC times for data request
        Histoic_start = utc_dt_open
        end_time = utc_dt_close
        print(f"Data request range: {Histoic_start} to {end_time}")

        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            start=Histoic_start,
            end=end_time,
            timeframe=TimeFrame(time_incriment, TimeFrameUnit.Minute)
        )

        try:
            historic = stock_client.get_stock_bars(request_params)
            historic = historic.df
            historic = historic.bfill()
            historic.reset_index(inplace=True)
            historic['timestamp'] = pd.to_datetime(historic['timestamp'], unit='ms', utc=True)
            historic = historic.set_index('timestamp')

            historic = historic.drop_duplicates(keep='last')
            historic = historic.pivot(columns='symbol', values=historic.columns)

            # ✅ Filter for market hours using the correct UTC times for this date
            is_open = (historic.index >= utc_dt_open) & (historic.index <= utc_dt_close)
            df_open = historic[is_open]

            print(f"📊 Data points for {historical_date}: {len(df_open)}")
            if len(df_open) > 0:
                print(f"   First timestamp: {df_open.index.min()}")
                print(f"   Last timestamp: {df_open.index.max()}")
                expected_minutes = (utc_dt_close - utc_dt_open).total_seconds() / 60
                print(f"   Expected minutes: {expected_minutes:.0f}")
                print(f"   Actual minutes: {len(df_open)}")
                print(f"   Completeness: {(len(df_open) / expected_minutes):.1%}")
            else:
                print(f"⚠️ No data found for {historical_date}")

            all_df_open.append(df_open)

        except Exception as e:
            print(f"❌ Error fetching data for {historical_date}: {e}")
            continue

# Combine all data
df_multiday = pd.concat(all_df_open)
df_multiday.sort_index(inplace=True)

# ✅ CRITICAL FIX: Remove duplicates after concatenation
print(f"📊 Before deduplication: {df_multiday.shape}")
df_multiday = df_multiday[~df_multiday.index.duplicated(keep='last')]
print(f"📊 After deduplication: {df_multiday.shape}")

# ✅ FIXED: Restructure to match cleaned_preprocessed.csv format with robust feature creation
def restructure_to_environment_format(df_multiday):
    """
    Restructure the multi-index data to match the environment's expected format
    where coin is a column and date is a column, similar to cleaned_preprocessed.csv
    """
    print("�� Restructuring data to environment format...")
    
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
                symbols.add(col[1])  # col[1] is the symbol name
        
        # Ensure we have valid symbols
        symbols = [s for s in symbols if s and s.strip()]
        
        # Create a row for each symbol
        for symbol in symbols:
            row_data = {'date': date_val, 'coin': symbol}  # Use 'coin' as column name from the start
            
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
    
    # Sort by date and symbol
    df_final = df_final.sort_values(['date', 'coin']).reset_index(drop=True)
    
    # ✅ CRITICAL: Check what columns actually exist in the data
    print("🔍 Analyzing available columns...")
    available_columns = df_final.columns.tolist()
    print(f"Available columns: {available_columns}")
    
    # Define expected base columns (case-insensitive check)
    expected_base_cols = ['high', 'low', 'open', 'close', 'volume', 'quoteVolume', 'weightedAverage']
    available_base_cols = []
    
    for expected_col in expected_base_cols:
        # Check for exact match first
        if expected_col in available_columns:
            available_base_cols.append(expected_col)
            print(f"✅ Found: {expected_col}")
        else:
            # Check for case variations
            found = False
            for col in available_columns:
                if col.lower() == expected_col.lower():
                    available_base_cols.append(col)  # Use the actual column name
                    print(f"✅ Found (case variation): {expected_col} -> {col}")
                    found = True
                    break
            if not found:
                print(f"❌ Missing: {expected_col}")
    
    print(f"\n📊 Available base columns: {available_base_cols}")
    
    # ✅ NORMALIZE: Scale down large values to match crypto scale
    print("\n📊 Normalizing data scale to match crypto data...")
    
    # # Normalize volume to crypto scale (divide by 10000 to get reasonable values)
    # if 'volume' in available_base_cols:
    #     df_final['volume'] = df_final['volume'] / 10000.0
    #     print("✅ Normalized volume (divided by 10000)")
    
    # Normalize prices to crypto scale (divide by 100 to get reasonable values)
    # price_cols = ['high', 'low', 'open', 'close']
    # for col in price_cols:
    #     if col in available_base_cols:
    #         df_final[col] = df_final[col] / 100.0
    #         print(f"✅ Normalized {col} (divided by 100)")
    
    # ✅ EXACT IMPLEMENTATION: Create features EXACTLY like pre_process.py
    print("\n📊 Creating features EXACTLY like pre_process.py...")
    
    # 1. Create volume-based features (vh, vl, vc) - RATIOS like pre_process.py
    if 'high' in available_base_cols and 'open' in available_base_cols:
        df_final["vh"] = df_final["high"] / df_final["open"]  # High/Open ratio
        print("✅ Created: vh (high/open ratio)")
    
    if 'low' in available_base_cols and 'open' in available_base_cols:
        df_final["vl"] = df_final["low"] / df_final["open"]   # Low/Open ratio
        print("✅ Created: vl (low/open ratio)")
    
    if 'close' in available_base_cols and 'open' in available_base_cols:
        df_final["vc"] = df_final["close"] / df_final["open"] # Close/Open ratio
        print("✅ Created: vc (close/open ratio)")
    
    # 2. ✅ FIXED: Create difference features using a safer approach
    print("\nCreating difference features (ABSOLUTE DIFFERENCES like pre_process.py)...")
    
    # ✅ FIXED: Create difference features with proper NaN handling
    print("\nCreating difference features (ABSOLUTE DIFFERENCES like pre_process.py)...")

    # Sort by symbol and date to ensure proper grouping
    df_final = df_final.sort_values(['coin', 'date']).reset_index(drop=True)

    # ✅ FIXED: Use a safer approach for creating difference features
    for coin in df_final['coin'].unique():
        coin_mask = df_final['coin'] == coin
        coin_data = df_final[coin_mask].copy()
        
        # Create difference features for this coin
        if 'open' in available_base_cols:
            differences = coin_data["open"] - coin_data["open"].shift(1)
            # ✅ FIX: Only fill the first NaN, not all NaN values
            differences.iloc[0] = 0.0  # Set first difference to 0 instead of NaN
            df_final.loc[coin_mask, "open_s"] = differences
        
        if 'volume' in available_base_cols:
            differences = coin_data["volume"] - coin_data["volume"].shift(1)
            differences.iloc[0] = 0.0
            df_final.loc[coin_mask, "volume_s"] = differences
        
        if 'quoteVolume' in available_base_cols:
            differences = coin_data["quoteVolume"] - coin_data["quoteVolume"].shift(1)
            differences.iloc[0] = 0.0
            df_final.loc[coin_mask, "quoteVolume_s"] = differences
        
        if 'weightedAverage' in available_base_cols:
            differences = coin_data["weightedAverage"] - coin_data["weightedAverage"].shift(1)
            differences.iloc[0] = 0.0
            df_final.loc[coin_mask, "weightedAverage_s"] = differences

    print("✅ Created difference features (absolute differences)")

    # ✅ FIXED: Create rolling averages with proper NaN handling
    # Build the list of available features for rolling averages
    available_features = []
    if 'vh' in df_final.columns:
        available_features.append('vh')
    if 'vl' in df_final.columns:
        available_features.append('vl')
    if 'vc' in df_final.columns:
        available_features.append('vc')
    if 'open_s' in df_final.columns:
        available_features.append('open_s')
    if 'volume_s' in df_final.columns:
        available_features.append('volume_s')
    if 'quoteVolume_s' in df_final.columns:
        available_features.append('quoteVolume_s')
    if 'weightedAverage_s' in df_final.columns:
        available_features.append('weightedAverage_s')

    ROLLING_WINDOWS = [7, 14, 30]  # Like pre_process.py

    print(f"\nCreating rolling averages for available features: {available_features}")

    # ✅ FIXED: Use a safer approach for rolling averages
    for coin in df_final['coin'].unique():
        coin_mask = df_final['coin'] == coin
        coin_data = df_final[coin_mask].copy()
        
        for col in available_features:
            for window in ROLLING_WINDOWS:
                # Calculate rolling average for this coin and feature
                rolling_avg = coin_data[col].rolling(window=window, min_periods=1).mean()
                # ✅ FIX: Use bfill() like in pre_process.py instead of fillna(0)
                rolling_avg = rolling_avg.bfill()
                df_final.loc[coin_mask, f"{col}_roll_{window}"] = rolling_avg

    # ✅ FIX: Only fill remaining NaN values, not all zeros
    df_final = df_final.fillna(method='bfill')#.fillna(0)
    
    # ✅ CRITICAL FIX: Ensure we only have 'coin' column, not 'symbol'
    if 'symbol' in df_final.columns:
        # If we have both 'coin' and 'symbol', drop 'symbol' and ensure 'coin' has the right values
        if 'coin' in df_final.columns:
            # Check if 'coin' column is empty and 'symbol' has values
            if df_final['coin'].isna().all() or df_final['coin'].eq('').all():
                df_final['coin'] = df_final['symbol']
            df_final = df_final.drop(columns=['symbol'])
        else:
            # If only 'symbol' exists, rename it to 'coin'
            df_final = df_final.rename(columns={'symbol': 'coin'})
    
    # ✅ CRITICAL FIX: Ensure we have the exact same columns as cleaned_preprocessed.csv
    # Remove any extra columns that shouldn't be there
    expected_columns = ['date', 'coin', 'high', 'low', 'open', 'close', 'volume', 'quoteVolume', 'weightedAverage']
    extra_columns_to_remove = []
    
    for col in df_final.columns:
        if col not in expected_columns and not col.startswith(('vh', 'vl', 'vc', 'open_s', 'volume_s', 'quoteVolume_s', 'weightedAverage_s')) and not col.endswith(('_roll_7', '_roll_14', '_roll_30')):
            extra_columns_to_remove.append(col)
    
    if extra_columns_to_remove:
        print(f"🗑️ Removing extra columns: {extra_columns_to_remove}")
        df_final = df_final.drop(columns=extra_columns_to_remove)
    
    # ✅ CRITICAL FIX: Ensure we have all required columns, create missing ones with default values
    if 'quoteVolume' not in df_final.columns:
        print("⚠️ Creating missing 'quoteVolume' column with default values")
        df_final['quoteVolume'] = df_final['volume']  # Use volume as default
    
    if 'weightedAverage' not in df_final.columns:
        print("⚠️ Creating missing 'weightedAverage' column with default values")
        df_final['weightedAverage'] = df_final['close']  # Use close as default
    
    # ✅ CRITICAL FIX: Reorder columns to match cleaned_preprocessed.csv exactly
    # First, get all the base columns in the right order
    base_columns = ['date', 'coin', 'high', 'low', 'open', 'close', 'volume', 'quoteVolume', 'weightedAverage']
    
    # Then get all the derived columns (vh, vl, vc, *_s, *_roll_*)
    derived_columns = [col for col in df_final.columns if col not in base_columns]
    
    # Sort derived columns to match the order in cleaned_preprocessed.csv
    derived_columns.sort()
    
    # Combine in the right order
    final_column_order = base_columns + derived_columns
    
    # Reorder the dataframe
    df_final = df_final[final_column_order]
    
    # ✅ CRITICAL FIX: Remove any duplicate coin/date combinations
    print(f"📊 Before coin/date deduplication: {df_final.shape}")
    df_final = df_final.drop_duplicates(subset=['date', 'coin'], keep='last')
    print(f"📊 After coin/date deduplication: {df_final.shape}")
    
    # Sort back by date and coin for consistency
    df_final = df_final.sort_values(['date', 'coin']).reset_index(drop=True)
    
    print(f"\n✅ Restructured data shape: {df_final.shape}")
    print(f"📅 Date range: {df_final['date'].min()} to {df_final['date'].max()}")
    print(f"�� Coins: {df_final['coin'].unique()}")
    
    # Verify we have all required features
    all_created_features = available_features.copy()
    for col in available_features:
        for window in ROLLING_WINDOWS:
            all_created_features.append(f"{col}_roll_{window}")
    
    missing_features = [f for f in all_created_features if f not in df_final.columns]
    if missing_features:
        print(f"⚠️ Missing features: {missing_features}")
    else:
        print(f"✅ All {len(all_created_features)} required features created!")
    
    print(f"\n📊 Final feature count: {len(all_created_features)} features per coin")
    print(f"📊 Total features: {len(all_created_features) * len(df_final['coin'].unique())}")
    
    print(f"\n�� Sample data:")
    print(df_final.head(10))
    
    return df_final

# Restructure the data
df_environment_format = restructure_to_environment_format(df_multiday)

# Save the restructured data
df_environment_format.to_csv("cleaned_preprocessed_restructured.csv", index=True)
print("�� Saved restructured data to 'cleaned_preprocessed_restructured.csv'")

# ✅ FIXED: Proper episode length calculation
print("\n🔍 DEBUGGING DATA STRUCTURE:")
print(f"Original df_multiday shape: {df_multiday.shape}")
print(f"Restructured data shape: {df_environment_format.shape}")
print(f"Unique dates: {df_environment_format['date'].nunique()}")
print(f"Unique coins: {df_environment_format['coin'].nunique()}")
print(f"Rows per coin per day: {df_environment_format.shape[0] / (df_environment_format['date'].nunique() * df_environment_format['coin'].nunique()):.1f}")

# Analyze trading day structure
print("\n📈 Analyzing trading day structure...")

# Ensure only 'symbol' is present (drop 'coin' if it slipped in earlier)
if 'coin' in df_environment_format.columns and 'coin' in df_environment_format.columns:
    df_environment_format = df_environment_format.drop(columns=['coin'])

# Steps per day should be the count of unique minute timestamps in [13:30, 20:00] per day,
# not multiplied by number of symbols. Count unique minutes per day.
df_environment_format['day'] = df_environment_format['date'].dt.date
df_environment_format['minute'] = df_environment_format['date'].dt.floor('min')

# Count unique minutes per day (deduplicate across symbols)
daily_counts = (
    df_environment_format[['day', 'minute']]
    .drop_duplicates()
    .groupby('day')
    .size()
)

print(f"📊 Trading days analyzed: {len(daily_counts)}")
print(f"📊 Average steps per day: {daily_counts.mean():.1f}")
print(f"📊 Median steps per day: {daily_counts.median():.1f}")
print(f"�� Min steps per day: {daily_counts.min()}")
print(f"�� Max steps per day: {daily_counts.max()}")
print(f"�� Standard deviation: {daily_counts.std():.1f}")

# Calculate expected steps based on time increment
# From your code: time_incriment = 1 (1-minute intervals)
expected_minutes = (datetime.datetime.combine(datetime.datetime.min, datetime.time(20, 0)) - 
                  datetime.datetime.combine(datetime.datetime.min, datetime.time(13, 30))).total_seconds() / 60
expected_steps = int(expected_minutes)  # Should be ~390 minutes (6.5 hours)

print(f"\n�� Expected steps per day (13:30-20:00): {expected_steps}")
print(f"📊 Average completeness: {(daily_counts.mean() / expected_steps):.1%}")

# Identify incomplete days
incomplete_days = daily_counts[daily_counts < 0.95 * expected_steps]
if len(incomplete_days) > 0:
    print(f"\n⚠️  Incomplete trading days (<95% complete): {len(incomplete_days)}")
    print("   These days may have missing data or early market closes")
    for day, steps in incomplete_days.head(5).items():
        completeness = steps / expected_steps
        print(f"   {day}: {steps}/{expected_steps} steps ({completeness:.1%})")

# Enhanced visualization
plt.figure(figsize=(15, 10))

# Plot 1: Steps per day distribution
plt.subplot(2, 2, 1)
plt.hist(daily_counts.values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(daily_counts.mean(), color='red', linestyle='--', label=f'Mean: {daily_counts.mean():.1f}')
plt.axvline(daily_counts.median(), color='orange', linestyle='--', label=f'Median: {daily_counts.median():.1f}')
plt.xlabel('Steps per Trading Day')
plt.ylabel('Frequency')
plt.title('Distribution of Steps per Trading Day')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Steps over time
plt.subplot(2, 2, 2)
plt.plot(daily_counts.index, daily_counts.values, marker='o', alpha=0.7)
plt.axhline(daily_counts.mean(), color='red', linestyle='--', label=f'Mean: {daily_counts.mean():.1f}')
plt.xlabel('Date')
plt.ylabel('Steps')
plt.title('Steps per Day Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Plot 3: Completeness distribution
completeness_ratios = daily_counts / expected_steps
plt.subplot(2, 2, 3)
plt.hist(completeness_ratios, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
plt.axvline(1.0, color='red', linestyle='--', label='100% Complete')
plt.xlabel('Completeness Ratio')
plt.ylabel('Frequency')
plt.title('Distribution of Day Completeness')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Box plot of steps
plt.subplot(2, 2, 4)
plt.boxplot(daily_counts.values)
plt.ylabel('Steps per Day')
plt.title('Box Plot of Steps per Day')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n🎯 Recommended episode length: {min(391, int(round(daily_counts.median())))} steps")
print(f"📊 Data ready for RL environment training!")

# Enhanced RL training analysis
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

# Additional RL training strategy analysis
print("\n" + "="*60)
print("🧠 RL TRAINING STRATEGY ANALYSIS")
print("="*60)

avg_steps = daily_counts.mean()
median_steps = daily_counts.median()

print("🤔 TRAINING APPROACH OPTIONS:")
print("\n1️⃣  FIXED DAY START APPROACH:")
print("   ✅ Pros:")
print("      • Consistent episode structure")
print("      • Agent learns market open patterns")
print("      • Easier to debug and reproduce")
print("      • Matches real trading behavior")
print("   ❌ Cons:")
print("      • Less data variety")
print("      • May overfit to market open patterns")
print("      • Limited exploration of mid-day scenarios")

print("\n2️⃣  RANDOM START POINT APPROACH:")
print("   ✅ Pros:")
print("      • Maximum data utilization")
print("      • Better exploration of different market conditions")
print("      • More diverse training scenarios")
print("      • Higher sample efficiency")
print("   ❌ Cons:")
print("      • Inconsistent episode lengths")
print("      • May learn unrealistic transitions")
print("      • Harder to interpret behavior")

print("\n3️⃣  HYBRID APPROACH (RECOMMENDED):")
print("   ✅ Best of both worlds:")
print("      • Use average day length as episode length")
print("      • Allow random start points within valid ranges")
print("      • Ensure minimum episode length for learning")
print("      • Add day boundary awareness")

print(f"\n�� SPECIFIC RECOMMENDATIONS:")
print(f"   • Episode length: {median_steps:.0f} steps (median)")
print(f"   • Minimum episode length: {daily_counts.quantile(0.1):.0f} steps (10th percentile)")
print(f"   • Maximum episode length: {daily_counts.quantile(0.9):.0f} steps (90th percentile)")

df_multiday = pd.read_csv('cleaned_preprocessed_restructured.csv')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(df_multiday.shape)

df_multiday.head(30)
