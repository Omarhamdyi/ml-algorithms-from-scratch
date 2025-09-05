import numpy as np

def normal_equation(X, y):
    '''
    Normal Equation for Linear Regression:
        W = (X^T X)^(-1) X^T y

    Dimensions:
        X:  (n_samples, n_features) -> (20640, 8)
        X.T:(n_features, n_samples) -> (8, 20640)
        y:  (n_samples,)            -> (20640,)
        W:  (n_features,)           -> (8,)

    Example with California housing:
        (X^T X): (8, 8)
        (X^T X)^(-1): (8, 8)
        (X^T y): (8,)
        Result W: (8,)
    '''
    # adding ones coulmn for the intercept
    X = np.hstack([np.ones((X.shape[0],1)),X])

    # safer to use pinv instead of inv
    W = np.linalg.pinv(X.T @ X) @ X.T @ y
    y_pred = X @ W 

    # computing cost function
    err = (y - y_pred)
    cost = (err.T @ err ) / (2 * len(y))
    return y_pred, W , cost
