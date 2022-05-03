import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib.pyplot as plt

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'])

    # remove missing temp samples
    df.dropna(subset=['Temp'], axis=0, inplace=True)

    # remove temp values lower than -20 or higher than 60
    df = df.loc[(df['Temp'] >= -20) & (df['Temp'] <= 60)]

    # Consider only samples which their features range correctly
    df = df.loc[(df['Year'] >= 0) & (df['Year'] <= 2022)]
    df = df.loc[(df['Month'] > 0) & (df['Month'] <= 12)]
    df = df.loc[(df['Day'] > 0) & (df['Day'] <= 31)]

    # create day of year feature and remove Date feature as it is already
    # represented in other features
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df.drop("Date", axis=1, inplace=True)

    # replace categorical features with label encoding
    dummies = pd.get_dummies(df.City)
    df = pd.concat([df, dummies], axis='columns')
    df.drop(['City'], axis='columns', inplace=True)

    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data(
        '/Users/97254/PycharmProjects/IML.HUJI/datasets/City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    israel_samples = df[df['Country'] == "Israel"]
    israel_samples["Year"] = df["Year"].astype(str)
    fig1 = px.scatter(israel_samples, x="DayOfYear", y="Temp", color="Year",
                     title="Temperature as a function of day of year")
    fig1.show()
    std = israel_samples.groupby('Month').agg(np.std)
    x_axis = ["January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"]
    y_axis = list(std['Temp'])
    fig2 = plt.figure(figsize=(15,5))
    plt.bar(x_axis, y_axis, color='maroon', width=0.4)
    plt.title("Standard deviation of daily temp for each month")
    plt.show()

    # Question 3 - Exploring differences between countries
    data_group_by_country = df.groupby(['Country', 'Month'],
                                       as_index=False).agg(
        {'Temp': ['mean', 'std']})
    fig3 = px.line(x=data_group_by_country['Month'],
                   y=data_group_by_country[('Temp', 'mean')],
                   color=data_group_by_country['Country'],
                   labels={'x': 'Month', 'y': 'Average Temperature',
                           'color': 'Country'},
                   error_y=data_group_by_country[('Temp', 'std')],
                   title='Average Temperature per Month')
    fig3.show()
    fig3.write_image('/Users/97254/Desktop/graphs/average_temp.png')

    # Question 4 - Fitting model for different values of `k`
    train = israel_samples.sample(frac=0.75, random_state=1)
    test = israel_samples.drop(train.index)
    train_y = train['Temp']
    train_X = train.drop(columns="Temp")
    test_y = test['Temp']
    test_X = test.drop(columns="Temp")

    k_list = [i for i in range(1, 11)]
    loss = []

    for k in k_list:
        polynomial_model = PolynomialFitting(k)
        fit = polynomial_model.fit(np.array(train_X['DayOfYear']),
                               np.array(train_y))
        polynomial_model.predict(np.array(test_X['DayOfYear']))
        cur_loss = round(polynomial_model.loss(np.array(test_X['DayOfYear']), np.array(test_y)), 2)
        loss.append(cur_loss)
        print("k: " + str(k) + ", loss: " + str(cur_loss))
    fig, ax = plt.subplots()
    ax.bar(k_list, loss, )
    ax.set_title("Test error recorded for each value of k")
    ax.set_xlabel("K - degree of polynomial model")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.show()


    # Question 5 - Evaluating fitted model on different countries
    chosen_k = 5
    polynomial_model = PolynomialFitting(chosen_k)
    fit = polynomial_model.fit(np.array(israel_samples['DayOfYear']),
                               np.array(israel_samples['Temp']))

    countries = {"South Africa": 0, "Jordan": 0, "The Netherlands": 0}
    for country in countries.keys():
        cur_country = df[df['Country'] == country]
        polynomial_model.predict(np.array(cur_country['DayOfYear']))
        countries[country] = round(polynomial_model.loss(np.array(cur_country['DayOfYear']),
                                               np.array(cur_country['Temp'])), 2)
    names = list(countries.keys())
    values = list(countries.values())
    plt.bar(range(len(countries)), values, tick_label=names)
    plt.title("Loss of a model fitted to Israel")
    plt.show()

