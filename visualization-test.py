# Python 2D plotting library
from matplotlib import style
# Pyplot provides a MATLAB-like plotting framework
import matplotlib.pyplot as plt
# Fundamental package for scientific computing
import numpy as np
# Python Data Analysis Library
import pandas as pd

# ggplot is a plotting system for Python based on R's ggplot2
style.use('ggplot')

pd.set_option('display.width', 1000)


def visualize_ticker(ticker):
    """Visualizing data and searching for patterns.
    
    """
    df = pd.read_csv('sp500-joined.csv', parse_dates=True, index_col=0)

    # 100 days rolling moving average
    df['{}_100MA'.format(ticker)] = df['{}_Close'.format(ticker)].rolling(window=100, min_periods=0).mean()

    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)

    ax1.plot(df.index, df['{}_Close'.format(ticker)])
    ax1.plot(df.index, df['{}_100MA'.format(ticker)])
    ax2.bar(df.index, df['{}_Volume'.format(ticker)])

    plt.show()


def visualize_tickers(*args):
    """Visualizing data and searching for patterns.
    Create linear plot graphs for several stocks 
    """
    df = pd.read_csv('sp500-joined.csv', parse_dates=True, index_col=0)

    tickers = [t for t in args]
    tickers_100ma, tickers_close, tickers_volume = [], [], []

    # 100 days rolling moving average
    for ticker in tickers:
        df['{}_100MA'.format(ticker)] = df['{}_Close'.format(ticker)].rolling(window=100, min_periods=0).mean()

        tickers_100ma.append('{}_100MA'.format(ticker))
        tickers_close.append('{}_Close'.format(ticker))

    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)

    ax1.plot(df.index, df[tickers_close])
    ax1.plot(df.index, df[tickers_100ma])

    plt.show()


def visualize_corr():
    """Visualizing data and searching for patterns.
    Building a correlation heatmap of S&P 500 stock prices.
    """
    df = pd.read_csv('sp500-joined-closes.csv')

    # Compute pairwise correlation of columns, excluding NA/null values
    df_corr = df.corr()
    df_corr.to_csv('sp500corr.csv')

    # Extract values from the corr table, and assign it as Numpy array
    data1 = df_corr.values

    # Define fig1 as matplotlib figure
    fig1 = plt.figure()

    # Add subplot in fig1, 111 means height, width, and plot number
    ax1 = fig1.add_subplot(111)

    # Set a color map of heatmap, pcolor: plot color, cmap: colormap
    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)

    # Add colorbar to the subplot
    fig1.colorbar(heatmap1)

    # Set ticks for x and y axis
    # Numpy.arange: return evenly spaced values within a given interval
    # 0.5 is the middle (0 to 1)
    # minor=False means disable minor ticks from axis
    ax1.set_xticks(np.arange(data1.shape[0]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[1]) + 0.5, minor=False)

    # Flip y axis for better readability
    ax1.invert_yaxis()

    # Move x axis to the top of subplot from the bottom
    ax1.xaxis.tick_top()

    # Assign labels, df_corr.columns and df_corr.index are identical
    column_labels = df_corr.columns
    row_labels = df_corr.index

    # Set tick labels for x and y axis
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)

    # Rotate x axis tick labels vertically
    plt.xticks(rotation=90)

    # Set the heatmap's color range from -1 to 1, clim: color limit
    heatmap1.set_clim(-1, 1)

    # Automatically adjust the subplot, so that fits into the figure
    plt.tight_layout()

    # Save the figure as a file
    # plt.savefig('correlations.png', dpi=300)

    # Display the figure
    plt.show()


# Visualize one stock with 100 days moving average
visualize_ticker('AMZN')

# Visualize Big 5 Tech. Companies
visualize_tickers('AAPL', 'GOOGL', 'GOOG', 'AMZN', 'FB', 'MSFT')

# Visualize heatmap of stock correlation
visualize_corr()
