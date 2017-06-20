import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


# Big 5 tech companies: Apple, Amazon, Facebook, Alphabet (Google), Microsoft
# Alphabet has two classes stocks (GOOG shares have no voting rights while GOOGL shares do).
big5 = ['AAPL', 'AMZN', 'FB', 'GOOG', 'GOOGL', 'MSFT']

df_sp500_closes = pd.read_csv('sp500-joined-closes.csv', parse_dates=True, index_col=0)
df_big5_closes = df_sp500_closes[big5]

# Create ALPB column for Alphabet, calculate mean of GOOG and GOOGL
df_big5_closes['ALPB'] = df_big5_closes[['GOOG', 'GOOGL']].mean(axis=1)
df_big5_closes.drop(['GOOG', 'GOOGL'], 1, inplace=True)

# Save to csv
df_big5_closes.to_csv('big5.csv')

# Exponentially-weighted Functions
fig, axes = plt.subplots(nrows=2, ncols=1, sharex='all', sharey='all')
ewm100 = df_big5_closes.ewm(span=100, min_periods=0, adjust=True, ignore_na=False).mean()

df_big5_closes.plot(ax=axes[0])
ewm100.plot(ax=axes[1])

axes[0].set_title('Big 5 Tech. Companies Stock Price')
axes[1].set_title('100 Days Exponentially-weighted Moving Average')
plt.xlabel('Time')
axes[0].set_ylabel('Stock Price')
axes[1].set_ylabel('Stock Price')


# Six-month return correlations to S&P 500
# SPX: S&P 500 index, downloaded manually from Yahoo Finance
spx = pd.read_csv('SPX.csv', parse_dates=True, index_col=0)
spx.rename(columns={'Adj Close': 'SPX'}, inplace=True)
spx.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

spx_rets = spx / spx.shift(1) - 1
returns = df_big5_closes.pct_change()

# Need to fix rolling_corr() -> DataFrame.rolling().corr(another DF)
df_corr = pd.rolling_corr(returns, spx_rets, 125, min_periods=100, pairwise=True)
df_corr.dropna()


big5_list = ['AAPL', 'AMZN', 'FB', 'MSFT', 'ALPB']
print('Average of six-month retun correlations to S&P 500:')
for b in big5_list:
    print(b, df_corr[f"{b}"].mean())

df_corr.plot()
plt.title('Six-month return correlations to S&P 500')
plt.xlabel('Time')
plt.ylabel('Correlation')
plt.show()
