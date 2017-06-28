# Counter is dict subclass for counting hashable objects
from collections import Counter
import numpy as np
import pandas as pd
# Open source machine learning library
from sklearn import svm, neighbors
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
# Calculating mathematical statistics of numeric (Real-valued) data
from statistics import mean

pd.set_option('display.width', 300)


def prep_data_for_labels(ticker):
    """Read S&P 500 close price data and manipulate and normalize for preprocessing.
    :param ticker: a stock symbol which will be predicted
    :return: tickers, df, days
    """
    # How many days in the future we need prices for
    days = 7

    # Read daily close prices of all stocks, index is 'Date'
    df = pd.read_csv('sp500-closes.csv', index_col=0)

    # Convert ticker column names into a list
    tickers = df.columns.values.tolist()

    # Replace NA/NaN values by 0
    df.fillna(0, inplace=True)

    # Normalization: add new columns of pct. change values for the next i days
    for i in range(1, days + 1):
        # Pct. change within next i days and today's price = (future price in i days - today's price) / today's price
        # Shift index by -i periods (shift up the column by i rows)
        df[f'{ticker}_{i}D'] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    # Replace NA/NaN values by 0
    df.fillna(0, inplace=True)

    return tickers, df, days


def label_classes(*args):
    """Target Labeling criteria
    :param args: every rows of f'{ticker}_{i}D' columns
    :return: 1, -1, or 0
    """

    # Put args into a list
    cols = [c for c in args]

    requirement = 0.005

    for col in cols:
        if col > requirement:
            return 1
        elif col < -requirement:
            return -1
        else:
            return 0


def extract_feature_target(ticker):
    """Mapping Target Labels
    :param ticker: a stock symbol
    :return: X, y, df
    """
    # Get return values from prep_manipulation function
    tickers, df, days = prep_data_for_labels(ticker)

    # Mapping labels to the target
    df[f'{ticker}_target'] = list(map(label_classes, *[df[f'{ticker}_{i}D'] for i in range(1, days + 1)]))

    vals = df[f'{ticker}_target'].values.tolist()

    # Change vals integer into string
    str_vals = [str(i) for i in vals]

    # Counts labels by classes
    print("Target Label Spread:", Counter(str_vals))

    # Cleaning data
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    # Percent change over given number of periods.
    df_vals = df[[ticker for ticker in tickers]].pct_change(periods=1)
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    # Features: daily pct. change of all S&P 500 stocks
    X = df_vals.values

    # Target: 1, 0, -1
    y = df[f'{ticker}_target'].values

    return X, y, df


def classification(ticker):
    """Classify a specific stock
    :param ticker: a stock symbol
    :return: accuracy
    """
    print(ticker)

    X, y, df = extract_feature_target(ticker)

    # Split train and test size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Classifiers
    clf1 = svm.LinearSVC()
    clf2 = neighbors.KNeighborsClassifier()
    clf3 = RandomForestClassifier()
    # n_jobs: The no. of jobs to run in parallel for fit.
    # If -1, then the number of jobs is set to the number of cores.
    eclf = VotingClassifier(estimators=[('lsvc', clf1), ('knc', clf2), ('rfc', clf3)], voting='hard', n_jobs=-1)

    # Principal Component Analysis (PCA)
    pca = PCA(n_components=10)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    pca.fit(X_test)
    X_test = pca.transform(X_test)

    # Cross-validation
    params = {}
    grid = GridSearchCV(estimator=eclf, param_grid=params, cv=10)
    grid = grid.fit(X_train, y_train)

    predictions = grid.predict(X_test)
    print("Predicted Label Spread:", Counter(predictions))

    accuracy = grid.score(X_test, y_test)
    print("Accuracy:", accuracy)

    return accuracy


def classification_all():
    """Classify all S&P 500 stocks."""
    df = pd.read_csv('sp500-closes.csv', index_col=0)
    tickers = df.columns.values.tolist()

    accuracies = []

    for count, ticker in enumerate(tickers):
        try:
            accuracy = classification(ticker)
            accuracies.append(accuracy)
            # print(f"{ticker} accuracy: {accuracy}")
            print(count)
        except:
            print(f"{ticker} has only one class in label")

    print(f"Average accuracy ({count} stocks): {mean(accuracies)}")


big5 = ['AAPL', 'AMZN', 'FB', 'GOOG', 'GOOGL', 'MSFT']
big5_accuracies = []

for b in big5:
    big5_accuracies.append(classification(f'{b}'))

print("Average Accuracy on Big 5:", mean(big5_accuracies))


# classification('AMZN')

# classification_all()
