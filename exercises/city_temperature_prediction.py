import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
import plotly.graph_objects as go

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

    data = pd.read_csv(filename, parse_dates=["Date"]).dropna().drop_duplicates()                                       #TODO:dont get the parse date thing

    # delete samples
    data = data[(0<data.Month) & (data.Month<=12)]
    data = data[(0<data.Day) & (data.Day<= 31)]
    data = data[(-10<=data.Temp)]

    # change date to the day of year
    data['DayOfYear'] = data["Date"].dt.dayofyear
    return data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data('/Users/matancohen/Local_docs/IML.HUJI/datasets/City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    israel_data = data[data['Country'] == "Israel"]
    px.scatter(israel_data, 'DayOfYear', 'Temp', color='Year', title='Daily AvgTemp as function of of Day of year').show()
    new_data = israel_data.groupby(['Month'])['Temp'].std().reset_index()
    px.bar(new_data,x = 'Month', y ='Temp', title = 'STD of Temperature for each month in year' ).show()

    # Question 3 - Exploring differences between countries
    grouped_data = data.groupby(['Country','Month'])['Temp'].agg(['mean', 'std']).reset_index(['Country','Month'])                  #TODO: understand multi index and why need reset index here/
    px.line(grouped_data, x = 'Month', y ='mean', error_y='std',color='Country', title= 'Avg temp (with STD) as a function of Month' ).show()

    # Question 4 - Fitting model for different values of `k`
    # note only 'DayOfYear is used to predict Temperature(Temp)
    train_X, train_y, test_X, test_y = np.asarray(split_train_test(israel_data['DayOfYear'], israel_data['Temp'], 0.75))
    total_test_errors = []
    for k in range(1,11):
        model = PolynomialFitting(k)
        model._fit(train_X,train_y)
        loss = np.round(model._loss(test_X,test_y),decimals=2)
        total_test_errors.append(loss)
    print(total_test_errors)
    px.bar(x = np.arange(1,11,1), y=total_test_errors, labels={'x': 'K- polynomial degree', 'y': 'Temperature loss'}, text_auto=True).show()




    # Question 5 - Evaluating fitted model on different countries
    # chose polynomial degree of k=5

    model = PolynomialFitting(k=5)
    model._fit(train_X,train_y)
    countries = data.Country.drop_duplicates().to_list()  #could have done --- data.Country.unique()
    countries.remove('Israel')
    losses = []
    for country in countries:
        country_data = data[data['Country']==country]
        loss = np.round(model._loss(country_data["DayOfYear"],country_data["Temp"]),decimals=2)
        losses.append(loss)
    px.bar(x = countries,y=losses,labels={'x': "countries",'y': "Temperature Loss"}).show()


