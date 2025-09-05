import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
def visualize(cost_history):
    plt.plot(list(range(len(cost_history))), cost_history, '-r')
    plt.xlabel('iterations')
    plt.ylabel("cost")
    plt.grid()
    plt.show()


def gradient_descent(X,y,step_size = 0.01 , precision = 0.00001 , max_iter = 1000):
    X = np.hstack([np.ones((X.shape[0] , 1)) , X])
    samples , features = X.shape

    # cost = 1/2n * sum(X*W - y_gt) **2
    def cost(X, y , weights):
        y_pred = X @ weights
        err = y_pred - y
        cost = (err.T @ err ) / (2 * samples)
        return cost

    # cost = 1/n * sum(X*W - y_gt) * X
    def deriv_cost(X,y,weights):
        y_pred = X @ weights
        err = y_pred - y
        gradient = (X.T @ err) / samples
        return gradient
    

    # initialize random wights 
    np.random.seed(0)
    cur_weights = np.random.rand(features)
    last_weights = np.full(features , np.inf)

    cost_history , weights_history = [] ,[]
    iter = 0
    while norm(cur_weights - last_weights) > precision and iter < max_iter:
        last_weights = cur_weights.copy()
        gradients = deriv_cost(X,y,cur_weights)
        cur_weights = cur_weights - step_size * gradients
        cost_history.append(cost(X,y,cur_weights))
        iter +=1

    visualize(cost_history)
    y_pred = X @ cur_weights
    
    return y_pred , cur_weights , cost(X,y,cur_weights)



