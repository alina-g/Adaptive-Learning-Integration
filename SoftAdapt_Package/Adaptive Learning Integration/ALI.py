#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 20:27:16 2020

@author: alinagataullina
"""

#########################  THE ONE FUNCTION DO IT ALL #########################   
    
"""
Adaptive Learning Integration Function: Simply calling this function will 

Usage : 
    
    input: 
            which_SA : which variation of SoftAdapt you would want to use
                SoftAdapt ->  the original (weaker) version
                Plush SoftAdapt -> the more robust (defult) variation of SA
                Downey SoftAdapt -> A very smoothed out version of SA. Use 
                    with caution and for very specific cases
                    
            AdaLearn_bool: Whether you want to use AdaLearn or not
            
            

"""
# beta to change is here!
import MakeArgs

class ALI():
    
    def __init__(self, recent_loss_tensor, which_SA="plush", AdaLearn_bool="False",\
                 beta=0.5, args = MakeArgs.make_args()):
        
        self.recent_loss_tensor = recent_loss_tensor
        self.which_SA = "plush"
        self.AdaLearn_bool = "False"
        self.beta = 0.5
        self.args = MakeArgs.make_args()

def ALI_func(self):
    
    # print all the parameters once at the very begining
    if args["global_count"] == 0 : 
      print("The Parameters of ALI : ");
      print("Soft Adapt Variation: {}".format(which_SA));
      print("Use AdaLearn: {}".format(AdaLearn_bool));      
      print("softmax Beta: {}".format(beta));
      print("learning rate : {}".format(args["lr"]))
      args["global_count"] += 1 ;
    


    # if we want to use AdaLearn
    if AdaLearn_bool == True : 
        
        # calculate the running average in an efficient matter
        avg_calc(recent_loss_tensor[0],recent_loss_tensor[1],args);
        
        # Set the Adaptive learning Rate
        # Decay the weight according to the performance 
        lr_decay(args);
            
    
    if len(recent_loss_tensor) == 2 :
        #store the last 5 losses for loss 1
        get_5loss(recent_loss_tensor[0].data.item(),0,args);
        get_5loss(recent_loss_tensor[1].data.item(),1,args);
                        
   
    elif len(recent_loss_tensor) == 3 :
        get_5loss(recent_loss_tensor[0].data.item(),0,args);
        get_5loss(recent_loss_tensor[1].data.item(),1,args);
        get_5loss(recent_loss_tensor[2].data.item(),2,args);
    
    else : 
        
        print("support for this coming soon");
    
    
    if args["global_count"] > 4 :
        
         if len(recent_loss_tensor) == 2 :
             slopes = np.zeros(2);
             slopes[0] = FD(args["loss1"],args);
             slopes[1] = FD(args["loss2"],args);
             
             set_hyper(alpha_assign(slopes,beta,recent_loss_tensor,which_SA)\
                       ,args,len(recent_loss_tensor));
             
        
         elif len(recent_loss_tensor) == 3 :
             slopes = np.zeros(3);
             slopes[0] = FD(args["loss1"],args);
             slopes[1] = FD(args["loss2"],args);
             slopes[2] = FD(args["loss3"],args);
                
             set_hyper(alpha_assign(slopes,beta,recent_loss_tensor,which_SA)\
                       ,args,len(recent_loss_tensor));
        
         else:
             
             print("Implementation coming soon : ) ")
                
        
# to see if we need to return alpha or not for the user
    #if hasattr(args, 'user_lazy'):
    if 1 < 2 :   
       alpha_return = np.zeros(len(recent_loss_tensor))
      
      
       # if user did not make the global dictionary
       #if args['user_lazy'] == 'y' : 
       if 1 < 2 : 
           # return the alphas
           alpha_return = args['alphas'];

       return alpha_return
    
    
    
    
