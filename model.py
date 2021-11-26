from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pandas as pd

def run_model(X_train,y_train,X_test,y_test):

    lin_model = LinearRegression()
    lin_model.fit(X_train,y_train)
    y_pred = lin_model.predict(X_test)
    test_mse = metrics.mean_squared_error(y_test, y_pred)
    return test_mse

