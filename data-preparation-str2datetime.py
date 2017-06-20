import pandas as pd

tickers = ['LMT', 'NBL', 'NWL']

df = pd.DataFrame()

for ticker in tickers:
    print(ticker)

    df = pd.read_csv(f"{ticker}.csv")
    print(df.head())

    # Change str to date type (e.g. 15-Nov-16  >>  2016-11-15)
    df['Date'] = pd.to_datetime(df['Date'])

    # Set index
    df.set_index('Date', inplace=True)

    # Sort by date
    df.sort_index(axis=0, ascending=True, inplace=True)

    df.to_csv(f"data/{ticker}.csv")
    print(df.head())
