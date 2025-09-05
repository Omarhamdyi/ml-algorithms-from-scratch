from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
def scikit_implementation(X,y):
    model = LinearRegression() # default: fit_intercept=True
    model.fit(X,y)

    y_pred = model.predict(X)
    
    W = np.r_[model.intercept_, model.coef_]

    cost = mean_squared_error(y_pred , y) / 2
    return y_pred , W , cost


