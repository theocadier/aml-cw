#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
print(os.getcwd())

# Dataset: 2010 to till date daily trends of NASDAQ-100 Stocks
df = pd.read_csv('NASDAQ100.csv', sep='\t', parse_dates=["Date"], index_col="Date")

# Basic EDA
print(df.columns)
print(df.dtypes)
print(df.index)

# Visual EDA
# 1) Plot historical closing prices over time
# Pivot so that each ticker name is a column of its own 'Adj Close'

pivoted = df.pivot_table(index="Date", columns='Name', values='Adj Close')

# Sort columns alphabetically so the order is consistent everywhere
pivoted = pivoted.reindex(sorted(pivoted.columns), axis=1)

# Plot all tickers’ Adj Close together (legend=False hides the overcrowded legend)
plt.figure(figsize=(12, 6))
pivoted.plot(legend=False)
plt.title("All NASDAQ-100 Companies: Historical Adjusted Close")
plt.xlabel("Date")
plt.ylabel("Adj Close (USD)")
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show();

# 2) Compute and plot a 30-day rolling moving average of Closing Price
rolling_window = 30
df['30D_MA'] = df['Adj Close'].rolling(window=rolling_window).mean()

# Compute the daily average across all tickers
market_avg = pivoted.mean(axis=1)

# 30-day moving average of that market-wide series
market_avg_ma30 = market_avg.rolling(window=30).mean()

plt.figure(figsize=(10,4))
market_avg.plot(label="Market Avg Adj Close", alpha=0.6)
market_avg_ma30.plot(label="30-Day MA of Market Avg", color="orange")
plt.title("NASDAQ-100 Average Adjusted Close vs 30-Day MA")
plt.legend()
plt.ylabel("Price (USD)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show();

# 3) Calculate daily returns (% returns) and plot distribution
df['Log_Returns'] = (df['Adj Close'] / df['Adj Close'].shift(1)).apply(lambda x: np.log(x))
df['Log_Returns'].dropna(inplace=True)

returns = df['Log_Returns'].dropna()
returns_pct = returns * 100

plt.figure(figsize=(8, 4))
returns_pct.hist(bins=10000, density=True, alpha=0.75)

plt.xlim(-5, 5)
plt.title('Histogram of Daily Log Returns (in %)')
plt.xlabel('Log Return (%)')
plt.ylabel('Density')
plt.grid(True)
plt.tight_layout()
plt.show();


# %%
