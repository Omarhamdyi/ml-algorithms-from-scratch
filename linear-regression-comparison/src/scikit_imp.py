import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def scikit_implementation(X,y):
    model = LinearRegression(fit_intercept=False)
    model.fit(X,y)

    y_pred = model.predict(X)

    cost = mean_squared_error(y_pred , y) / 2
    r2 = r2_score(y, y_pred)
    return y_pred , model.coef_ , cost , r2


