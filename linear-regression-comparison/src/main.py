from  load_data import load_data_scaled 
from normal_equation import normal_equation
from scikit_imp import scikit_implementation
from gradient_descent import gradient_descent
from visualization import visualize_iter
import numpy as np

if __name__ == "__main__":
    X , y = load_data_scaled()
    y_pred_neq , W_neq , cost_neq , r2_neq = normal_equation(X,y)
    y_pred_sk , W_sk  , cost_sk ,r2_sk  = scikit_implementation(X,y)
    y_pred_gd , W_gd  , cost_gd ,r2_gd , cost_history = gradient_descent(X,y,step_size = 0.1 , max_iter = 10000)

    print(f'At Normal Equation \n the weights ={W_neq}\n Cost(MSE) ={cost_neq} and R_squared = {r2_neq}\n')
    print(f'At Scikit Learn \n the weights ={W_sk}\n Cost(MSE) ={cost_sk} and R_squared = {r2_sk}\n')
    print(f'At Gradeint Descent \n the weights ={W_gd}\n Cost(MSE) ={cost_gd} and R_squared = {r2_gd}\n')

    visualize_iter(cost_history)
    



'''
    At Normal Equation 
    the weights =[  3.72961166   6.33214009   0.48122468 -15.13916237  21.76021606
    -0.1418736   -4.70531325  -3.96456829  -4.3625181 ]
    Cost(MSE) =0.2621604930923035 and R_squared = 0.6062326851998052

    At Scikit Learn 
    the weights =[  3.72961166   6.33214009   0.48122468 -15.13916237  21.76021606
    -0.1418736   -4.70531325  -3.96456829  -4.3625181 ]
    Cost(MSE) =0.26216049309230355 and R_squared = 0.6062326851998051

    At Gradeint Descent
    the weights =[ 3.46904075  5.608185    0.58193581  0.51505836  0.97072581  0.48714267
    0.07316374 -3.65842993 -3.94199966]
    Cost(MSE) =0.27042315429105385 and R_squared = 0.5938220970331505
'''
