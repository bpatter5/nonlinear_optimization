# =============================================================================
# Created By: Brett Patterson
# Updated By: Brett Patterson
# Created On: 9/20/2018
# Updated On: 9/20/2018
# Purpose: Perform unconstrained gradient descent on convex functions
# =============================================================================

# if you want to calc the Hessian you'll need to use
# from autograd import hessian
# and call it twice on your function

from autograd import grad
from autograd import numpy as np
from nonlinear_optimization.search import backtrack_line_search as bls

def example_function(x):
    y = x[0]**2 + 3 * x[1]**2 + 2*x[0]*x[1] + x[0] + 3
    return(y)
    

def gradient_descent(x, f_x, grad_f_x, epsilon=.000001):
    delta_x = -1 * grad_f_x(x)
    
    while(np.linalg.norm(delta_x)>epsilon):
        t = bls.backtracking_line_search(delta_x=delta_x, f_x=f_x, x=x)
        
        x = x + t * delta_x
        delta_x = -1 * grad_f_x(x)
        
    return(x)
    
    
grad_f_x = grad(example_function)
x = np.array([0.75,2.0])

x_star = np.array([-0.75, 0.25])
x_opt_grad = gradient_descent(x, example_function, grad_f_x)

print(x_star)
print(x_opt_grad)
print(np.sum((x_star - x_opt_grad)**2))