

"""
Alpha assignment : a function that calls one of the variations of SoftAdapt
and it will return the appropiate values for each alpha

Usage : 
    
    Argument : A vector of slopes n
               A constant value for the softmax called kappa (default1 1) 
               A tensor of loss values at each iteration loss_tensor 
                   e.g. if your loss function is MSE + l1, then 
                   loss_tensor = [MSE, l1];
               A string indicating which method you want to use
               (default should be PlushAdapt)


"""

import numpy as np
import SoftAdapt

class Alpha_Assign():
    
    def __init__(self, n, kappa, loss_tensor, string, alpha, args, loss_num):
        
        self.n = n
        self.kappa = kappa
        self.loss_tensor = loss_tensor
        self.string = string
        self.alpha = alpha
        self.args = args
        self.loss_num = loss_num
    

    def alpha_assign(self):

   
        self.alpha = np.zeros(len(self.n));
  
        if self.string == "plush":
            for i in range(len(self.n)):
                    self.alpha[i] = SoftAdapt.PlushAdapt(self.n, self.kappa, self.loss_tensor, i);
                
        if self.string == "downy":
            for i in range(len(self.n)):
                    self.alpha[i] = SoftAdapt.DownyAdapt(self.n, self.kappa, self.loss_tensor, i);

        if self.string == "soft":
            for i in range(len(self.n)):
                    self.alpha[i] = SoftAdapt.SoftAdapt(self.n, self.kappa, self.loss_tensor, i);
    
    
        return self.alpha 
    
    
    """
Set Hyper :
    sets the hyper parameters alpha0 through alpha n
    
    Usage : 
        
        takes in the vector Alpha, a dictionary of arguments (highly recommend
        a global dictionary) and number of loss functions
        outputs all the values of 
        
        
    Caution : 
        
        alpha -> the vector returned by alpha_assign() 
        alphas -> a global array that its entries are multiplied by the loss
        
"""


    def set_hyper(self):
    
        for i in range (0, self.loss_num) : 
   
            self.args["alphas"][i] = self.alpha[i];
            
            
            
            
            
            
            