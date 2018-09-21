# =============================================================================
# Created By: Brett Patterson
# Updated By: Brett Patterson
# Created On: 9/20/2018
# Updated On: 9/20/2018
# Purpose: Perform backtracking line search to support optimization methods
# =============================================================================

from autograd import numpy as np

def backtracking_line_search(delta_x, f_x, x):
    alpha = np.random.uniform(0.0, 0.5)
    beta = np.random.uniform()
    t = 1
    
    f_x_t = f_x(x + t * delta_x)
    f_x_alpha = f_x(x) + (alpha * t * (np.dot(delta_x, delta_x)))
    
    while(f_x_t > f_x_alpha):
        
        t = beta * t
        
        f_x_t = f_x(x + t * delta_x)
        f_x_alpha = f_x(x) + (alpha * t * (np.dot(delta_x, delta_x)))
        
    return(t)
    