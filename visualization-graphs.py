# Python 2D plotting library
from matplotlib import style
# Pyplot provides a MATLAB-like plotting framework
import matplotlib.pyplot as plt
# Python Data Analysis Library
import pandas as pd

# ggplot is a plotting system for Python based on R's ggplot2
style.use('ggplot')
pd.set_option('display.width', 1000)


def visualize_ticker(ticker):
    """Visualizing a stock with moving average
    """
    df = pd.read_csv('sp500-joined.csv', parse_dates=True, index_col=0)

    # 100 days rolling moving average
    df[f"{ticker}_100MA"] = df[f"{ticker}_Close"].rolling(window=100, min_periods=0).mean()

    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)

    ax1.plot(df.index, df[f"{ticker}_Close"])
    ax1.plot(df.index, df[f"{ticker}_100MA"])
    ax2.bar(df.index, df[f"{ticker}_Volume"])

    ax1.set_ylabel('Stock Price')
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Time')
    ax1.set_title(f"{ticker} Stock Price and 100 Days Moving Average")
    ax1.set_xticklabels([])

    plt.show()


def visualize_tickers(*args):
    """Visualizing multiples stocks.
    """
    df = pd.read_csv('sp500-joined-closes.csv', parse_dates=True, index_col=0)

    tickers = [t for t in args]
    tickers_100ma, tickers_close, tickers_volume = [], [], []

    # 100 days rolling moving average
    for ticker in tickers:
        df[f"{ticker}_100MA"] = df[f"{ticker}"].rolling(window=100, min_periods=0).mean()
        tickers_100ma.append(f"{ticker}_100MA")
        tickers_close.append(f"{ticker}")

    plt.plot(df.index, df[tickers_close])
    plt.plot(df.index, df[tickers_100ma])

    plt.legend([c for c in tickers_close])
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Multiple Stock Prices and 100 Days Moving Averages')
    plt.show()


# Visualize one stock with 100 days moving average
# visualize_ticker('AMZN')

# Visualize multiple stocks
# visualize_tickers('AAPL', 'AMZN')
