from collections import Counter
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn import svm, neighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from statistics import mean

pd.set_option('display.width', 1000)


def prep_manipulation(ticker):
    """Read S&P 500 close price data and manipulate and normalize for preprocessing
    """
    # How many days in the future we need prices for
    days = 5
    # Read daily close prices of all stocks, index is 'Date'
    df = pd.read_csv('sp500-joined-closes.csv', index_col=0)
    # Convert ticker column names into a list
    tickers = df.columns.values.tolist()
    # Replace NA/NaN values by 0
    df.fillna(0, inplace=True)

    # Normalization: add new columns of pct. change values for the next i days
    for i in range(1, days + 1):
        # Pct. change within next i days and today's price = (price in i days - today's price) / today's price
        # Shift index by -i periods (shift up the column by i rows)
        df['{}_{}D'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    # Replace NA/NaN values by 0
    df.fillna(0, inplace=True)

    return tickers, df, days


def label_classes(*args):
    """Target Labeling criteria
    """
    cols = [c for c in args]
    requirement = 0.02

    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1

    return 0


def prep_map_labels(ticker):
    """Mapping Target Labels
    """
    tickers, df, days = prep_manipulation(ticker)

    df['{}_target'.format(ticker)] = \
        list(map(label_classes, *[df['{}_{}D'.format(ticker, i)] for i in range(1, days + 1)]))

    # print(df['{}_target'.format(ticker)])

    vals = df['{}_target'.format(ticker)].values.tolist()
    # Change vals integer into string
    str_vals = [str(i) for i in vals]
    print('Target Label Counts:', Counter(str_vals))

    # Cleaning data
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    # Percent change over given number of periods.
    df_vals = df[[ticker for ticker in tickers]].pct_change(periods=1)
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    # Features: daily pct. change of all S&P 500 stocks (may need to reduce no. of features, PCA analysis or clustering)
    X = df_vals.values
    # Target:
    y = df['{}_target'.format(ticker)].values

    return X, y, df


def classification(ticker):
    """Classify a specific stock
    """
    print(ticker)

    X, y, df = prep_map_labels(ticker)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf1 = svm.LinearSVC()
    clf2 = neighbors.KNeighborsClassifier()
    clf3 = RandomForestClassifier()
    # n_jobs: The no. of jobs to run in parallel for fit. If -1, then the number of jobs is set to the number of cores.
    # I don't see the difference after applying n_jobs, what's the problem?
    eclf = VotingClassifier(estimators=[('lsvc', clf1), ('knc', clf2), ('rfc', clf3)], voting='hard', n_jobs=-1)

    params = {}

    """
    #PCA
    pca = PCA(n_components=10)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    pca.fit(X_test)
    X_test = pca.transform(X_test)
    """

    grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)
    grid = grid.fit(X_train, y_train)

    """
    tscv = TimeSeriesSplit(n_splits=3)
    for train_index, test_index in tscv.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    """

    # eclf.fit(X_train, y_train)
    # use for loop for testing each classifiers

    predictions = grid.predict(X_test)
    print('Predicted Target Label Counts:', Counter(predictions))

    accuracy = grid.score(X_test, y_test)
    print('Accuracy:', accuracy)

    return accuracy

    # need to add precision-recall curve
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html


def classification_all():
    """Classify all S&P 500 stocks 
    """
    df = pd.read_csv('sp500-joined-closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    print(tickers)

    accuracies = []

    for count, ticker in tickers:
        try:
            accuracy = classification(ticker)
            accuracies.append(accuracy)
            # print("{} accuracy: {}".format(ticker, accuracy))
            print(count)
        except:
            print("{} has only one class in label".format(ticker))

    print("Average accuracy ({} stocks): {}".format(count, mean(accuracies)))


start = datetime.now()
classification('AMZN')
print(datetime.now() - start)

# classification_all()  # need to set n_jobs params
