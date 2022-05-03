from numpy import std

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)

    # remove missing price samples
    df.dropna(subset=['price'], axis=0, inplace=True)

    # replace numeric features with no value with the average value of the
    # column
    numeric = df.select_dtypes(include=np.number)
    numeric_columns = numeric.columns
    df[numeric_columns] = df[numeric_columns].fillna(df.mean())

    # make features range correctly
    df = df.loc[(df['view'] >= 0) & (df['view'] <= 4)]
    df = df.loc[(df['grade'] > 0) & (df['view'] <= 13)]
    df = df.loc[(df['condition'] > 0) & (df['view'] <= 5)]

    # remove id,longitude,latitude, sqft_living15, sqft_lot15, zipcode features as
    # they're irrelevant
    df.drop("id", axis=1, inplace=True)
    df.drop("long", axis=1, inplace=True)
    df.drop("lat", axis=1, inplace=True)
    df.drop("sqft_living15", axis=1, inplace=True)
    df.drop("sqft_lot15", axis=1, inplace=True)
    df.drop("zipcode", axis=1, inplace=True)

    # remove samples with non-postive features or non-integer number of
    # bathrooms
    df = df[df.select_dtypes(include=[np.number]).ge(0).all(1)]
    df['bathrooms int'] = df['bathrooms'].astype(int)
    df = df.loc[(df['bathrooms int'] == df['bathrooms'])]
    df.drop("bathrooms int", axis=1, inplace=True)


    # replace categorical features with label encoding
    df['date'] = pd.Series(df['date'].T).str.slice(stop=4)
    df['date'] = df['date'].astype('category')
    df['sale year'] = df['date'].cat.codes
    df.drop(labels='date', axis=1, inplace=True)
    df['waterfront'] = df['waterfront'].astype('category')
    df['waterfront'] = df['waterfront'].cat.codes

    # TODO: remove before submission
    # df.to_csv('C:/Users/97254/Desktop/output.csv')
    # print(df.head(10))
    # # print(df.isna().sum())
    # # print("============================")
    # print(df.dtypes)
    # # print("============================")
    #
    # # print("============================")

    response = df['price']
    df.drop("price", axis=1, inplace=True)
    return df, response


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in X.columns:
        rho = np.cov(X[feature], y)[0, 1] / (np.std(X[feature]) * np.std(y))
        fig = px.scatter(x=X[feature], y=y,
                         labels={'x': f'{feature}', 'y': 'Response'},
                         title='feature: ' + feature + ', Correlation: ' + str(
                             rho))
        fig.write_image(output_path + '/' + feature + ".png")


if __name__ == '__main__':
    print(10 / 100)
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    x, y = load_data(
        '/Users/97254/PycharmProjects/IML.HUJI/datasets/house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(x, y, '/Users/97254/Desktop/graphs')

    # Question 3 - Split samples into training- and testing sets.
    train_X, test_X, train_y, test_y = split_train_test(x, y,
                                                        train_proportion=.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    p_list = [i for i in range(10, 101)]
    loss = []
    confidence_interval = []
    for p in range(10, 101):
        cur_loss = []
        for i in range(10):
            cur_train_x = train_X.sample(frac=p / 100, random_state=i)
            cur_train_y = train_y.sample(frac=p / 100, random_state=i)
            linear_model = LinearRegression(True)
            fit = linear_model.fit(np.array(cur_train_x),
                                   np.array(cur_train_y))
            linear_model.predict(np.array(test_X))
            cur_loss.append(
                linear_model.loss(np.array(test_X), np.array(test_y)))
        loss.append(np.mean(cur_loss))
        confidence_interval.append(np.std(cur_loss))
    loss = np.array(loss)
    confidence_interval = np.array(confidence_interval)
    fig = go.Figure(layout=go.Layout(
        title_text='Mean loss of the linear model over increasing precentages of the test set'
        , xaxis={"title": 'Percentage'}, yaxis={"title": 'MSE over test set'}))
    fig.add_traces(
        go.Scatter(x=p_list, y=loss, mode="markers+lines", name='Mean Of MSE',
                   marker=dict(color="blue", opacity=.7)))
    fig.add_traces(
        go.Scatter(x=p_list, y=loss - 2 * confidence_interval, fill=None, mode='lines',
                   line=dict(color="lightgrey"), showlegend=False))
    fig.add_traces(
        go.Scatter(x=p_list, y=loss + 2 * confidence_interval, fill='tonexty', mode='lines',
                   line=dict(color="lightgrey"), showlegend=False))
    fig.show()