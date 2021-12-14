from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import pandas as pd
import plotly.graph_objects as pgo
import sys


def min_squares(train_data, test_data, norm):
    train_X = train_data.drop('score', axis=1)
    train_Y = train_data['score']
    min_square = LinearRegression().fit(train_X, train_Y)
    prediction = min_square.predict(test_data.drop('score', axis = 1))
    mse = mean_squared_error(test_data['score'], prediction)
    chart = pgo.Figure(data=[pgo.Table(header=dict(values=['Metodo', 'Normalizado','MSE', 'rating','fav_genre','cast','advertising','length']),
                 cells=dict(values=['Min Squares', norm, mse, min_square.coef_[0], min_square.coef_[1], min_square.coef_[2], min_square.coef_[3], min_square.coef_[4]]))])
    chart.show()


#def lasso_l1(train_data, test_data, norm):
#    train_X = train_data.drop('score', axis=1)
#    train_Y = train_data['score']
#    lasso = Lasso().fit(train_X, train_Y)
#    prediction = lasso.predict(test_data.drop('score', axis = 1))
#    mse = mean_squared_error(test_data['score'], prediction)
#    chart = pgo.Figure(data=[pgo.Table(header=dict(values=['Metodo', 'Normalizado','MSE', 'rating','fav_genre','cast','advertising','length']),
#                 cells=dict(values=['Lasso', norm, mse, lasso.coef_[0], lasso.coef_[1], lasso.coef_[2], lasso.coef_[3], lasso.coef_[4]]))])
#    chart.show()


def Normalizar(data):
    scaler = StandardScaler().fit(data)
    index = range(0, len(data.index))
    column = ['rating','fav_genre','cast','advertising','length','score']
    data = scaler.transform(data)
    df = pd.DataFrame(data, index = index, columns = column) 
    return df


def main(train, test):
    train_data = pd.read_csv(train)
    test_data = pd.read_csv(test)
    min_squares(train_data, test_data, 'No')
#    lasso_l1(train_data, test_data, 'No')
    ntrain_data = Normalizar(train_data)
    ntest_data = Normalizar(test_data)
    min_squares(ntrain_data, ntest_data, 'Si')
#    lasso_l1(ntrain_data, ntest_data, 'Si')

if __name__ == "__main__":
    train = sys.argv[1]
    test = sys.argv[-1]
    main(train, test)