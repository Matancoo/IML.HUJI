from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
from IMLearn.metrics.loss_functions import mean_square_error
from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"
# import matplotlib.pyplot as plt



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
    #load data and remove missing data
    data = pd.read_csv(filename).dropna().drop_duplicates()

    # keeping relevant feature columns
    # data = data.loc[:, ['date', 'bathrooms', 'sqft_basement','sqft_living', 'sqft_above', 'grade','price','bedrooms', 'yr_built',   'view','waterfront','condition']]
    data = data.loc[:, ['bathrooms', 'sqft_basement','sqft_living', 'sqft_above', 'grade','price','bedrooms', 'yr_built']]



    #checking value range (see kaggle)
    data = data[data.yr_built >= 1900]
    data = data[data.bathrooms > 0]      #found propreties with zero bathrooms
    data = data[data.price > 0]
    data = data[data.index != 15870]    # sample which has 33 bedrooms and 1.75 bathrooms which isnt logicl

    #feature engineering
    # data['month'] = data.date.apply(func = lambda x: x[:][4:6])
    # data['date'] = data.date.apply(func = lambda x: x[:][4:4])  #new feature date in years instead of --/--/----



    #turning categorical features to One-Hot vectors

    # data = pd.get_dummies(data, columns=['date'])
    # data = pd.get_dummies(data, columns=['view'])
    # data = pd.get_dummies(data, columns=['waterfront'])
    data = pd.get_dummies(data, columns=['grade'])
    # data = pd.get_dummies(data, columns=['condition'])








    # run on categories of categrical features and plot how they do with price (mean)
    # feature selection
    # I want to plot the mean average per year for example
    # price_mean = X.groupby(['yr_built'])['price'].mean()
    # price_mean.plot(kind='bar')
    # plt.show()



    # TODO: the features that I kept have a high correlation between each other

    y = data.pop('price')
    return (data, y)


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
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
        feature_col = X.loc[:, feature]

        cov_ = np.cov(feature_col, y)[0][1]
        std_feature = np.std(feature_col)
        std_response = np.std(y)
        corr = cov_ / (std_feature * std_response)


        y_axis = y
        x_axis = feature_col
        fig = go.Figure([go.Scatter(x=x_axis, y=y_axis, name='correlation' + str(corr), showlegend=True,
                                    # why need name here if i have title down there?
                                    marker=dict(color="black", opacity=.7), mode="markers",
                                    line=dict(color="black", width=0.1))],
                        layout=go.Layout(title=r"$\text{(1) what do i witite here? } $",
                                         xaxis={"title": "x - Price"},
                                         yaxis={"title": feature},
                                         height=400))
        fig.write_image('/Users/matancohen/Local_docs/feature_plots/' + feature + '.png')






# helper functions

def correlation_matrix(X: pd.DataFrame):
    # returns a correlation matrix of each feature with all others.
    data = X.select_dtypes(include=np.number)  # select only numerical features
    corr_matrix = data.corr(method='pearson').loc[:, ['price']].sort_values('price')
    corr_target = abs(corr_matrix)
    return corr_target



if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of housing prices dataset
    data, response = load_data("/Users/matancohen/Local_docs/IML.HUJI/datasets/house_prices.csv")


    # Question 2 - Feature evaluation with respect to response

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = np.asarray(split_train_test(data, response, 0.75))

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    scores = []
    for percent in range(10, 101):
        p_scores = []
        for i in range(10):
            train_set = pd.concat([train_X, train_y], axis=1).sample(frac=percent / 100)
            new_train_y = train_set.pop('price')
            new_train_x = train_set

            model = LinearRegression()
            model.fit(np.asarray(new_train_x), np.asarray(new_train_y))
            # calculating loss between prediction and response
            loss = model.loss(test_X, test_y)
            p_scores.append(loss)
        scores.append(p_scores)
    scores = np.asarray(scores)
    x_axis = np.arange(10, 101)
    means = scores.mean(axis=1)
    variances = scores.std(ddof=1, axis=1)

    go.Figure([go.Scatter(x=x_axis, y=means - 2 * variances, fill=None, mode="lines", line=dict(color="lightgrey"),
                          showlegend=False),
               go.Scatter(x=x_axis, y=means + 2 * variances, fill='tonexty', mode="lines", line=dict(color="lightgrey"),
                          showlegend=False),
               go.Scatter(x=x_axis, y=means, mode="markers+lines", marker=dict(color="black", size=1),
                          showlegend=False)],
              layout=go.Layout(title=r"$\text{(7) mean loss over 10 samples as a function of training set percentage}$",
                               xaxis={"title": "training set size (percentage)"},
                               yaxis={"title": "average loss"},
                               height=600)).show()






















    #____method 1___
    # categ_features = X.select_dtypes(include=['object']).columns
    # numeric_features = X.select_dtypes(exclude=['object']).columns
    # # print(categ_features)
    # print(numeric_features)


    # print(data.columns)
    # print(correlation_matrix(X))
    #
    # # TODO: is grade a
    # print(data.describe())
    # print(X.loc[:,'price'])


    # fig = go.Figure([go.Scatter(x=x_axis, y=y_axis, name='correlation' + str(corr), showlegend=True,
    #                             # why need name here if i have title down there?
    #                             marker=dict(color="black", opacity=.7), mode="markers",
    #                             line=dict(color="black", width=0.1))],
    #                 layout=go.Layout(title=r"$\text{(1) what do i witite here? } $",
    #                                  xaxis={"title": "x - Price"},
    # #TODO:   review this code


#
























#_________________PRACTICE_____________________________________________________
# # print(students_df[["lunch", "reading score"]])
# # print(students_df["reading score"].pow(2))
# #TODO: how to get median of specific colum?
# print(students_df["reading score"].median())

# selecting row by condition !!!!!
# print(students_df[students_df["reading score"]>reading_score_median].shape)
# print(students_df.shape)
#
#

#
# df = pd.DataFrame(np.array([np.random.choice(["Zohar", "Shelly", "Omer", "Avi"], 50), np.random.choice(["Linearit", "Intro", "Infi", "Probabilistic"], 50), np.random.randint(80, 101, 50)]).transpose(), columns=['Name', 'Course', 'Grade'])
# df["Grade"] = df["Grade"].astype(int)
#
# print("\n\nStudents df")
# print(df)
#
# print("\n\nCalculate average by student and by course")
# # print(df.groupby(['Name']).mean().reset_index())  #groupby names only uning mean
# print(df.groupby(by = ['Name', 'Course']).mean().reset_index()) #groupby maes and course using mean

# sorting
# print('\nsorting df')
# print(df.sort_values(by='Grade'))
# print(df.sort_index(), inplace = False)


# ------CREATE DATAFRAME________________
# tt = pd.DataFrame([('tom',200),('matan',100),('philip',500)], index=['worst','ok','best'], columns=('people','strength'))
# print(tt)
# print(tt.reset_index())
# print(tt.groupby(['strength'])) #TODO: what does this do?
# print(tt.groupby(['strength'],axis=1)) TODO: and this ?
# -------


# applying fucntions to colums
# df2 = pd.DataFrame(np.random.randint(1,10,(10,3)),columns=['col1','col2','col3'])
# print('\ndf2')
# print(df2)
#
# px.bar(data_frame=df2,
#        x= )
# #
# print('\ncalculate diff between min and max of each col')
# # print(df2.apply(func=np.sqrt, axis='index'))#TODO: when axis= 'index  it means we apply on the colums!!
# print(df2.apply(func= lambda x: x.max()-x.min())) #TODO: apply() works on entire rows/cols while applymap() on individual values
# print(df2.applymap(func=lambda x: x*100,))

# _______________ITERATE_______
#
# print('\n\n iterate over columns')
# for key,val in df2.iteritems():
#     print(key,val)
#
#
# print('\n\n iterate over rows')
# for row_idx,row in df2.iterrows():
#     print(row_idx,row)
