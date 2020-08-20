##################### ada learn ########################
#################### avg_calc ############################
"""
average calculator : given the most recent loss it will calculate the average with resepct
to the average in a memory efficient way (I think)

input : the most recent loss (for each individual loss)
outputs : it changes the global variable args.loss1_avg (and for all other losses) and does
not return a value



"""

import numpy as np

class AdaptiveLearning():
    
    def __ini__(self, recent_loss1, args):
        
        self.recent_loss1 = recent_loss1
        self.args = args

def avg_calc(self):
  
    if self.args["loss1_global_count"] > 4 :
        
        if self.args["loss1_avg"] == 0 : 
            
            self.args["loss1_avg"] = np.mean(self.args["loss2"]);
        
        self.args["loss1_avg"] = (self.args["loss1_avg"] * 5 + self.recent_loss1)/6;
    
    
    
    


################## adapt_lr ##########################
"""
adaptive learning : it checks a very simple criterion and if it is true then it will set 
the learning rate to the max, and it will change a global flag and iter whcih then are used
to decay back down 

input : the most recent losses 

output : it will update the following global variables : 
        args.lr
        args.flag
        args.adapt_iter2

"""

def adapt_lr(recent_loss1,args):
    
        
    if recent_loss1 > args["loss1_avg"] and args["flag"] == 'GO':
        args["lr"] = args["lr_max"];        
        args["flag"] = 'NO'
        args["adapt_iter2"] += 1;

#         print("lr is inc to max");
        
    if recent_loss1 > args["loss1_avg"] and args["flag"] == 'NO' and  args["adapt_iter2"] > 2:
        
        args["lr"] = args["lr_min"];        
        args["flag"] = 'NO2'
        args["adapt_iter"] += 1;

#         print("lr is dec to min");
          
          
        
   
################## lr_decay ##########################
"""
learning rate decay : it will increase or decrease the learning rate depending on the specific conditions

input : a dictionary of global variables args 

output : it will update the following global variables : 
        args.lr
        args.flag
        args.adapt_iter
"""

        
def lr_decay(args):

    if args["flag"] == 'NO2' and args["lr"] < args["lr_max"] : 
        print("OK AT LEAST IT GOES THROUGH IT!")
        args["lr"] *= 2;
        if args["lr"] > args["lr_max"] :
            
                args["lr"] = args["lr_max"];
        
        
    elif args["flag"] == 'NO' and args["lr"] > args["lr_min"] : 
  
        if args["lr"] > args["lr_min"]: 

            args["lr"] = args["lr_max"] * np.exp(-1 * args["adapt_iter"] * args["kappa"]);

        if args["lr"] < args["lr_min"] : 

            args["lr"] = args["lr_min"];

        else :

            args["flag"] = 'GO'     
    else:
        
        args["flag"] = 'GO'
    