#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
print(os.getcwd())

# Dataset: 2010 to till date daily trends of NASDAQ-100 Stocks
df = pd.read_csv('NASDAQ100.csv', sep='\t', parse_dates=[0], index_col=0)

# Basic EDA
print(df.columns)
print(df.dtypes)
print(df.index.dtype)

# Visual EDA
# 1) Plot historical closing prices over time
plt.figure(figsize=(10, 4))
df['Adj Close'].plot(title='Historical Closing Prices')
plt.ylabel('Close Price')
plt.grid(True)
plt.tight_layout()
plt.show();

# 2) Compute and plot a 30-day rolling moving average of Close
rolling_window = 30
df['30D_MA'] = df['Adj Close'].rolling(window=rolling_window).mean()

plt.figure(figsize=(10, 4))
df['Adj Close'].plot(label='Closing Price', alpha=0.6)
df['30D_MA'].plot(label=f'{rolling_window}-Day MA', color='orange')
plt.title(f'Close Price vs {rolling_window}-Day Moving Average')
plt.legend()
plt.ylabel('Price')
plt.grid(True)
plt.tight_layout()
plt.show();

# 3) Plot trading volume over time
plt.figure(figsize=(10, 3))
df['Volume'].plot(title='Trading Volume Over Time', color='green')
plt.ylabel('Volume')
plt.grid(True)
plt.tight_layout()
plt.show();

# 4) Calculate daily returns (% returns) and plot distribution
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
