# Beautiful Soup 4: pulling data from HTML or XML.
import bs4 as bs
# The pickle module implements binary protocols for serializing and de-serializing a Python object structure.
import pickle
# HTTP library
import requests
# Miscellaneous operating system interfaces
import os
# Basic date and time types
import datetime as dt
# Remote data access for pandas
import pandas_datareader.data as web
# Python Data Analysis Library
import pandas as pd

pd.set_option('display.width', 1000)



def save_sp500_tickers():
    """Step 1. Data Selection
    Getting tickers from S&P 500 companies, referencing the S&P 500 list from wikipedia.
    A ticker symbol is an abbreviation used to uniquely identify a particular stock on a stock market.
    The S&P 500 stock market index, maintained by S&P Dow Jones Indices,
    comprises 505 common stocks issued by 500 large-cap companies and traded on American stock exchanges,
    and covers about 80 percent of the American equity market by capitalization.
    """
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    # Convert the html source code into python objects, lxml is a parser.
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    # Find 'wikitable sortable' class in the HTML source code.
    table = soup.find('table', {'class': 'wikitable sortable'})
    # Define a list for saving S&P 500 ticker.
    tickers = []

    # Start from 1 to ignore the table header. 'tr' means table row.
    for row in table.findAll('tr')[1:]:
        # First column is the ticker of companies. 'td' means table data.
        ticker = row.findAll('td')[0].text
        # Append each ticker to the 'tickers' list.
        tickers.append(ticker)

    # Pickle the 'sp500tickers' list using the highest protocol available.
    # Open a pickle file in binary mode: “wb” to write it, and “rb” to read it.
    with open('sp500tickers.pickle', 'wb') as f:
        # Write a pickled representation of obj to the open file object file.
        pickle.dump(tickers, f)

    # print(tickers)

    return tickers


def get_data_from_google(reload_sp500=False):
    """Step 2-1. Data Preprocessing - Integration
    Gathering all company's, listed in the S&P 500, stock price data from Google finance.
    Downloaded 502 lists of stock price data out of 505 common stocks.
    Need to fix an error: couldn't access LMT, NWL, and NBL data.
    pandas_datareader._utils.RemoteDataError: Unable to read URL:
    http://www.google.com/finance/historical?q=LMT&startdate=Jan+01%2C+2000&enddate=Dec+31%2C+2016&output=csv
    """
    if reload_sp500:
        # Re-pull the S&P 500 list when reload_sp500=True
        tickers = save_sp500_tickers()
    else:
        with open('sp500tickers.pickle', 'rb') as f:
            tickers = pickle.load(f)

    if not os.path.exists('data'):
        os.mkdir('data')

    # Set start and end date range for pulling data
    start = dt.datetime(2000, 1, 1)
    end = dt.datetime(2017, 5, 29)

    for ticker in tickers:
        # 'ticker' is an items in 'tickers' list
        print(ticker)
        try:
            # 'format' performs a string formatting operation
            if not os.path.exists('data/{}.csv'.format(ticker)):
                df = web.DataReader(ticker, 'google', start, end)
                df.to_csv('data/{}.csv'.format(ticker))
            else:
                print("Already have {}.csv".format(ticker))
        except:
            print("Can't load {} data.".format(ticker))


def compile_data():
    """Step 2-2. Data Preprocessing - Integration / Cleaning
    In this step, multiple csv files merged into one main csv file.
    Compiling all the stock price into one data frame, and removing unnecessary columns.
    """
    with open('sp500tickers.pickle', 'rb') as f:
        tickers = pickle.load(f)

    # DataFrame is 2D size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows & columns)
    # Create csv for daily close values per ticker
    main_df = pd.DataFrame()

    # 'enumerate' is a built-in function, it returns an enumerate object
    for count, ticker in enumerate(tickers):
        try:
            df = pd.read_csv('data/{}.csv'.format(ticker))
            # Set the DataFrame index (row labels) using one or more existing columns. By default yields a new object.
            # The inplace parameter is for modifying the DataFrame in place (do not create a new object)
            # Set 'Date' as an index key
            df.set_index('Date', inplace=True)
            # 'columns' parameter: dict-like or functions are transformations to apply to that axis’ values
            df.rename(columns={'Close': ticker}, inplace=True)
            # Drop requested axis without creating a new object (0: row axis, 1: columns axis)
            df.drop(['Open', 'High', 'Low', 'Volume'], 1, inplace=True)
            if main_df.empty:
                main_df = df
            else:
                # 'how' is a methods of join {‘left’, ‘right’, ‘outer’, ‘inner’}, default: ‘left’
                main_df = main_df.join(df, how='outer')
        except:
            pass
        if count % 10 == 0:
            print(count)
    print(main_df.tail())
    main_df.to_csv('sp500-joined-closes.csv')

    # Create sp500 joined csv file, including daily closes, High-Low pct. change, and Open-Close pct. change
    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        try:
            df = pd.read_csv('data/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
            # Daily High-Low percentage difference per ticker
            df['{}_HL_pct_diff'.format(ticker)] = (df['High'] - df['Low']) / df['Low']
            # Daily Close-Open percentage change per ticker
            df['{}_daily_pct_chg'.format(ticker)] = (df['Close'] - df['Open']) / df['Open']
            df.rename(columns={'Close': '{}_Close'.format(ticker)}, inplace=True)
            df.rename(columns={'Volume': '{}_Volume'.format(ticker)}, inplace=True)
            df.drop(['Open', 'High', 'Low'], 1, inplace=True)
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
        except:
            pass
        if count % 10 == 0:
            print(count)
    print(main_df.tail())
    main_df.to_csv('sp500-joined.csv')

    """
    # Create csv containing daily HL percentage change per ticker
    main_df = pd.DataFrame()
    for count, ticker in enumerate(tickers):
        try:
            df = pd.read_csv('data/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
            df[ticker] = (df['High'] - df['Low']) / df['Low']
            df.drop(['Open', 'High', 'Low', 'Volume', 'Close'], 1, inplace=True)
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
        except:
            pass
        if count % 10 == 0:
            print(count)
    print(main_df.tail())
    main_df.to_csv('sp500-joined-hl-pct-chg.csv')

    # Create csv containing daily percentage change of Open/Close price 
    main_df = pd.DataFrame()
    for count, ticker in enumerate(tickers):
        try:
            df = pd.read_csv('data/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
            df[ticker] = (df['Close'] - df['Open']) / df['Open']
            df.drop(['Open', 'High', 'Low', 'Volume', 'Close'], 1, inplace=True)
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
        except:
            pass
        if count % 10 == 0:
            print(count)
    print(main_df.tail())
    main_df.to_csv('sp500-joined-daily-pct-chg.csv')
    """

# save_sp500_tickers()
# get_data_from_google()
# compile_data()
# visualize_corr()
