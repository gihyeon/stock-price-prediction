# 1. Data Preparation

# Beautiful Soup 4: pulling data from HTML or XML
import bs4 as bs
# Basic date and time types
import datetime as dt
# Miscellaneous operating system interfaces
import os
# Python Data Analysis Library
import pandas as pd
# Remote data access for pandas
import pandas_datareader.data as web
# The pickle module implements binary protocols for serializing and de-serializing a Python object structure
import pickle
# HTTP library
import requests

# Set an width of the display in characters
pd.set_option('display.width', 300)
# Set a maximum rows to display
pd.set_option('display.max_rows', 15)
# Set a maximum columns to display
pd.set_option('display.max_columns', 10)


def save_sp500_tickers():
    """Getting tickers from a "list of S&P 500 companies", Wikipedia.
    Saving the tickers list into a pickle file.
    :return: give tickers list to get_data_from_google(reload_sp500=True)
    """
    # Get the list webpage and assign it resp
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')

    # Convert the html source code into python objects using lxml parser
    soup = bs.BeautifulSoup(resp.text, 'lxml')

    # Find 'wikitable sortable' class in the HTML source code
    table = soup.find('table', {'class': 'wikitable sortable'})

    # Define a list for saving the S&P 500 tickers
    tickers = []

    # Get S&P 500 component tickers from the table, and append it to tickers list
    # Start from 1 to ignore the table header
    # tr (table row), td (table data)
    for row in table.findAll('tr')[1:]:
        # First column is the ticker symbol of companies.
        ticker = row.findAll('td')[0].text

        # Append each ticker to the 'tickers' list
        tickers.append(ticker)

    # Save the tickers list using the highest protocol available
    # Open a pickle file in binary mode: 'wb' to write it, and 'rb' to read it
    with open('sp500tickers.pickle', 'wb') as f:
        # Write a pickled representation of obj.
        pickle.dump(tickers, f)

    return tickers


def get_data_from_google(reload_sp500=False):
    """Gathering S&P 500 component stock price data from Google finance.
    Saving each stock price data as csv file.
    :param reload_sp500: boolean
    :return: none
    """
    # Regathering S&P 500 list when reload_sp500=True
    if reload_sp500:
        tickers = save_sp500_tickers()

    # Otherwise, load the saved list
    else:
        with open('sp500tickers.pickle', 'rb') as f:
            tickers = pickle.load(f)

    # Make a 'data' directory if it doesn't exist
    if not os.path.exists('data'):
        os.mkdir('data')

    # Set a date range for gathering data
    start = dt.datetime(2007, 6, 1)
    end = dt.datetime(2017, 5, 31)

    # Iterate for getting and saving stock price data by ticker
    for ticker in tickers:
        # 'ticker' is an item in 'tickers' list
        print(ticker)

        try:
            # f-string literal (f"{}" or f'{}') performs a string formatting operation
            if not os.path.exists(f'data/{ticker}.csv'):
                # Gather stock price data from Google Finance using Pandas DataReader
                df = web.DataReader(ticker, 'google', start, end)

                # Save the dataframe as csv file
                df.to_csv(f'data/{ticker}.csv')

            else:
                print(f"Already have {ticker}.csv")

        except:
            print(f"Can't load {ticker} data.")


def convert_format(*args):
    """Converting a format of data.
    :param args: ticker symbols
    :return: none
    """
    # Get tickers from args and save it to a list
    tickers = [t for t in args]

    for ticker in tickers:
        df = pd.read_csv(f'raw/{ticker}.csv')

        # Change str to date type (e.g. 15-Nov-16  >>  2016-11-15)
        df['Date'] = pd.to_datetime(df['Date'])

        # Set the DataFrame index (row labels) using one or more existing columns
        # By default, set_index() yields a new object.
        # The inplace parameter is for modifying the DataFrame in place (doesn't create a new object).
        # Set 'Date' as an index key
        df.set_index('Date', inplace=True)

        # Sort date in ascending order (default)
        df.sort_index(axis=0, inplace=True)

        df.to_csv(f'data/{ticker}.csv')

        print(ticker)
        print(df, end='\n\n')


def compile_data():
    """Merging multiple csv files merged into a main csv file.
    Compiling all the stock price data into one data frame, and removing unnecessary columns.
    :return: none
    """
    with open('sp500tickers.pickle', 'rb') as f:
        tickers = pickle.load(f)

    # DataFrame is 2D size-mutable,
    # potentially heterogeneous tabular data structure with labeled axes (rows & columns)
    main_df_closes, main_df_cnv = pd.DataFrame(), pd.DataFrame()

    # Create csv for daily close values per ticker
    # 'enumerate' built-in function returns an enumerate object one by one.
    for count, ticker in enumerate(tickers):
        try:
            df = pd.read_csv(f'data/{ticker}.csv')
            df.set_index('Date', inplace=True)

            # 'columns' parameter: dict-like or functions are transformations to apply to that axis’ values
            df.rename(columns={'Close': ticker}, inplace=True)

            # Drop requested axis without creating a new object (0: row axis, 1: columns axis)
            df.drop(['Open', 'High', 'Low', 'Volume'], 1, inplace=True)

            # Joining DataFrames
            if main_df_closes.empty:
                main_df_closes = df
            else:
                # 'how' is a methods of join {‘left’, ‘right’, ‘outer’, ‘inner’}, default: ‘left’
                main_df_closes = main_df_closes.join(df, how='outer')
        except:
            pass

        if count % 10 == 0:
            print(count)

    print(main_df_closes)
    main_df_closes.to_csv('sp500-closes.csv')

    # Create csv for daily close price and trade volume
    for count, ticker in enumerate(tickers):
        try:
            df = pd.read_csv(f'data/{ticker}.csv')
            df.set_index('Date', inplace=True)
            df.rename(columns={'Close': f'{ticker}_Close'}, inplace=True)
            df.rename(columns={'Volume': f'{ticker}_Volume'}, inplace=True)
            df.drop(['Open', 'High', 'Low'], 1, inplace=True)
            if main_df_cnv.empty:
                main_df_cnv = df
            else:
                main_df_cnv = main_df_cnv.join(df, how='outer')
        except:
            pass

        if count % 10 == 0:
            print(count)

    print(main_df_cnv)
    main_df_cnv.to_csv('sp500-closes-volumes.csv')


# save_sp500_tickers()
# get_data_from_google()
# convert_data('LMT', 'NBL', 'NWL')
# compile_data()
