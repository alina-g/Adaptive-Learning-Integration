# if the user does not want to have a seperate file, they can make all the-
                 # needed parameters here with this function
                 
                 
import np                 
                 
def make_args(*args):    
    
    args = {
        
        "alphas": np.zeros(2),
        "lr":0.001,
        "lr_max" : 0.01,
        "lr_min" : 0.0001,
        "loss1" : np.zeros(5),
        "loss2" : np.zeros(5),
        "global_count" : 0,
        "loss1_global_count" : 0,
        "loss2_global_count" : 0,
        "loss1_avg" : 0,
        "loss2_avg" : 0, 
        "flag": 'Go',
        "adapt_iter": 1,
        "adapt_iter2": 1,    
        "kappa": 1.5,
        "fd_order": 5,
        "num_epochs" : 30,
        "user_lazy": 'y'
    
     }    
    
    return args
    
    
   