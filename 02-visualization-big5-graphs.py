import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd

mpl.rcParams['figure.figsize'] = (15, 10)
style.use('ggplot')


# Big 5 Tech. Giants: Apple, Amazon, Facebook, Alphabet (Google), Microsoft
# Alphabet has two classes stocks (GOOG shares have no voting rights while GOOGL shares do).
big5 = ['AAPL', 'AMZN', 'FB', 'GOOG', 'GOOGL', 'MSFT']

df = pd.read_csv('sp500-closes.csv', parse_dates=True, index_col=0)
df_big5 = df[big5]

# Create ALPB column for Alphabet, calculate mean of GOOG and GOOGL
df_big5['ALPB'] = df_big5[['GOOG', 'GOOGL']].mean(axis=1)
df_big5.drop(['GOOG', 'GOOGL'], 1, inplace=True)
df_big5.sort_index(axis=1, inplace=True)

# Save to csv
df_big5.to_csv('big5.csv')

# Exponentially-weighted Functions
ewm100 = df_big5.ewm(span=100, min_periods=0, adjust=True, ignore_na=False).mean()

plt.plot(df_big5.index, df_big5)
plt.plot(df_big5.index, ewm100)

plt.title("Big 5 Tech. Giants' Stock Price and 100 Days Exponentially-weighted Moving Average")
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend(df_big5)


# Six-month return correlations to S&P 500
# SPX: S&P 500 index, downloaded manually from Yahoo Finance
spx = pd.read_csv('raw/SPX.csv', parse_dates=True, index_col=0)
spx.rename(columns={'Adj Close': 'SPX'}, inplace=True)
spx.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

spx_rets = spx / spx.shift(1) - 1
returns = df_big5.pct_change()

# Need to fix rolling_corr() -> DataFrame.rolling().corr(another DF)
corr = pd.rolling_corr(returns, spx_rets, 125, min_periods=100, pairwise=True)
corr.dropna()

big5_list = list(df_big5.columns.values)
print('Average of six-month retun correlations to S&P 500:')
for b in big5_list:
    print(b, corr[f'{b}'].mean())

corr.plot()
plt.title('Six-month return correlations to S&P 500')
plt.xlabel('Time')
plt.ylabel('Correlation')
plt.show()
