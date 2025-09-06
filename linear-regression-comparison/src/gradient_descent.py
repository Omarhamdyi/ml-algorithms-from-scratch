import numpy as np
from numpy.linalg import norm
from sklearn.metrics import r2_score

def gradient_check(weights , cost , deriv_cost):
    gradients = deriv_cost(weights)

    c = 1e-6
    for i in range(len(weights)):
        weights[i] += c
        cost1 = cost(weights[i]
                     )
        weights[i] -= (2*c)
        cost2 = cost(weights[i])

        gradient1 = gradients[i]
        gradient2 = cost1 - cost2 / (2*c)

        if not np.close(gradient1 , gradient2 , atol = 0.001):
            print(f'you have a problem in your deriv function')
            return False
        
    return True


def gradient_descent(X,y,step_size = 0.01 , precision = 0.00001 , max_iter = 1000):
    samples , features = X.shape
    # initialize random wights 
    np.random.seed(0)
    cur_weights = np.random.rand(features)
    last_weights = np.full(features , np.inf)

    # cost = 1/2n * sum(X*W - y_gt) **2
    def cost(weights):
        y_pred = X @ weights
        err = y_pred - y
        cost = (err.T @ err ) / (2 * samples)
        return cost

    # cost = 1/n * sum(X*W - y_gt) * X
    def deriv_cost(weights):
        y_pred = X @ weights
        err = y_pred - y
        gradient = (X.T @ err) / samples
        return gradient
    
    # assert gradient_check(cur_weights, cost, deriv_cost)


    cost_history , weights_history = [] ,[]
    iter = 0
    while norm(cur_weights - last_weights) > precision and iter < max_iter:
        last_weights = cur_weights.copy()
        gradients = deriv_cost(cur_weights)
        cur_weights = cur_weights - step_size * gradients
        cost_history.append(cost(cur_weights))
        iter +=1

    y_pred = X @ cur_weights
    r2 = r2_score(y, y_pred)

    
    return y_pred , cur_weights , cost(cur_weights) , r2 , cost_history



