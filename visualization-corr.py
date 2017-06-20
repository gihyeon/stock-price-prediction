import pandas as pd
# Fundamental package for scientific computing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

def visualize_corr(*args):
    """Visualizing a heatmap of stock correlation and searching for patterns.
    """
    tickers = [t for t in args]
    df = pd.read_csv('sp500-joined-closes.csv')

    # Compute pairwise correlation of columns, excluding NA/null values
    df_corr = df[tickers].corr()
    print("Correlation Table:")
    print(df_corr)
    df_corr.to_csv('big5-corr.csv')

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
    plt.savefig('correlations.png', dpi=300)

    # Display the figure
    plt.show()

# Visualize a correlation of top 10 tech companies
visualize_corr('AAPL', 'AMZN', 'FB', 'GOOG', 'GOOGL', 'MSFT',
               'IBM', 'CSCO', 'INTC', 'ORCL', 'V')
