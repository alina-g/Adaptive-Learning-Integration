#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 23:52:38 2019

@author: aliheydari
@email: aliheydari@ucdavis.edu
@web: https://www.ali-heydari.com

"""
###################### ADAPTIVE LEARNING INTEGRATION ##########################
################ Adaptive Loss Functions WITHOUT AdaLearn #####################
import os
os.system("pip install easydict");

import easydict
import numpy as np
import torch



version = "0.0.5"
backend = "PyTorch"



def Welcome_BEARD():
  
      print("\__________     __________/")
      print(" |         |-^-|         |")
      print(" |         |   |         |")
      print("  `._____.´     `._____.´")
      print("  \                     /")
      print("   \\\                 // ")
      print("    \\\    ////\\\\\\\   //")
      print("     \\\\\           /// ")
      print("       \\\\\\\\\\\\|////// ")
      print("         \\\\\\\\|//// ")


      print(" ")
      print("ALI {} for {} imported succsessfuly".format(version,backend))

    
Welcome_BEARD();

    
""" 
SOFTADAPT in various forms


Modified version of softmax with a little spice from beta

Beta is a hyperparameter that will either sharpen or dampen the peaks 

Default Beta is 1

Variations : 
    
******* SoftAdapt : A vanilla softmax with the value of the loss function 
        at each iteration multiplied by the exponent, i.e. 
        
        softAdapt(f,s) = f_i * e^(beta*s) / (sum f_je^(beta*s_j))
        where f is the loss value and s is the slope : 


******* PlushAdapt : The same idea as soft Adapt except the slopes and the loss
        function values are normalized at each iteration
        
        
        
******* DownyAdapt : Same as SoftAdapt except only the slopes are being normalized
        at each iteration       
        



Usage : 
    pass in a np vector n with a weight beta (if not sure what to use, then pass 1)
    returns softmax in the same dimensions as n

"""


# ################## SOFT ADAPT CLASS ######################################


class SoftAdapt():
	
	def __init__(self,n,beta,loss_tensor,i):
        self.n = n
        self.beta = beta
        self.loss_tensor = loss_tensor
        self.i = i

### Soft Adapt ###

	def SoftAdapt(self):
     # numerator
    
	#      self.n = -1 * self.n;
      
        if len(self.n) == 2 : 
     
        	fe_x = np.zeros(2);
         	fe_x[0] = self.loss_tensor[0].data.item() * np.exp(self.beta * (self.n[0] - np.max(self.n)));
         	fe_x[1] = self.loss_tensor[1].data.item() * np.exp(self.beta * (self.n[1] - np.max(self.n)));
         	denom = fe_x[0] + fe_x[1];

     	elif len(self.n) == 3 :
         
     	    fe_x = np.zeros(3);
        	fe_x[0] = self.loss_tensor[0].data.item() * np.exp(self.beta * (self.n[0] - np.max(self.n)));
         	fe_x[1] = self.loss_tensor[1].data.item() * np.exp(self.beta * (self.n[1] - np.max(self.n)));
         	fe_x[2] = self.loss_tensor[2].data.item() * np.exp(self.beta * (self.n[2] - np.max(self.n)));
         	denom = fe_x[0] + fe_x[1] + fe_x[2];  
                                               
     	else :
         	print("As of now, we only support 2 or 3 losses, please check input")

                                  
     	return (fe_x[self.i]/ denom)

### PlushAdapt ###

    def PlushAdapt(self):

        #   n = -1 * n;

        if len(self.n) == 2 : 
            fe_x = np.zeros(2);
         
         # Normalize the slopes!!!!
            self.n[0] = self.n[0] / (np.linalg.norm(self.n,1) + 1e-8);
            self.n[1] = self.n[1] / (np.linalg.norm(self.n,1) + 1e-8);
         
         # normalize the loss functions 
         
            denom2 = self.loss_tensor[0].data.item() + self.loss_tensor[1] 
    
            fe_x[0] = self.loss_tensor[0].data.item() / denom2;
            fe_x[1] = self.loss_tensor[1].data.item() / denom2;
        
            fe_x[0] = fe_x[0] * np.exp(self.beta * (self.n[0] - np.max(self.n)));
            fe_x[1] = fe_x[1] * np.exp(self.beta * (self.n[1] - np.max(self.n)));
  
            denom = fe_x[0] + fe_x[1];                                      
         
             return (fe_x[self.i]/ denom)
         


        elif len(self.n) == 3 :
         
            fe_x = np.zeros(3);
         
         # Normalize the slopes
             self.n[0] = self.n[0] / (np.linalg.norm(self.n,1) + 1e-8);
             self.n[1] = self.n[1] / (np.linalg.norm(self.n,1) + 1e-8);
             self.n[2] = self.n[2] / (np.linalg.norm(self.n,1) + 1e-8);

         
         # Normalize the loss functions          
            denom2 = self.loss_tensor[0].data.item() + self.loss_tensor[1].data.item() + self.loss_tensor[3].data.item() 
    
            fe_x[0] = self.loss_tensor[0].data.item() / denom2;
            fe_x[1] = self.loss_tensor[1].data.item() / denom2;
            fe_x[2] = self.loss_tensor[2].data.item() / denom2;

        
            fe_x[0] = fe_x[0] * np.exp(self.beta * (self.n[0] - np.max(self.n)));
            fe_x[1] = fe_x[1] * np.exp(self.beta * (self.n[1] - np.max(self.n)));
            fe_x[2] = fe_x[2] * np.exp(self.beta * (self.n[2] - np.max(self.n)));
  
            denom = fe_x[0] + fe_x[1] + fe_x[2] ;                                      
         
         
             return (fe_x[self.i]/ denom)
                                               
        else :
         
            print("As of now, we only support 2 or 3 losses, please check input")




### DOWNY SOFT ADAPT ###
         
    def DownyAdapt(self):
        
    # numerator
        fe_x = np.zeros(2);
        self.n[0] = self.n[0] / (np.linalg.norm(self.n,1) + 1e-8);
        self.n[1] = self.n[1] / (np.linalg.norm(self.n,1) + 1e-8);
    
        denom2 = self.loss_tensor[0].data.item() + self.loss_tensor[1] 
    
        fe_x[0] = self.loss_tensor[0].data.item() / denom2;
        fe_x[1] = self.loss_tensor[1].data.item() / denom2;
    
        fe_x[0] = fe_x[0] * np.exp(self.beta * (self.n[0] - np.max(self.n)));
        fe_x[1] = fe_x[1] * np.exp(self.beta * (self.n[1] - np.max(self.n)));
  
        denom = fe_x[0] + fe_x[1];    

        if len(self.n) == 2 : 
     
            fe_x = np.zeros(2);
         
            # Normalize the slopes
            self.n[0] = self.n[0] / (np.linalg.norm(self.n,1) + 1e-8);
            self.n[1] = self.n[1] / (np.linalg.norm(self.n,1) + 1e-8);
         
            fe_x[0] = self.loss_tensor[0].data.item() * np.exp(self.beta * (self.n[0] - np.max(self.n)));
            fe_x[1] = self.loss_tensor[1].data.item() * np.exp(self.beta * (self.n[1] - np.max(self.n)));
         
            denom = fe_x[0] + fe_x[1];
         
            return (fe_x[self.i]/ denom)


        elif len(self.n) == 3 :
         
            fe_x = np.zeros(3);
            self.n[0] = self.n[0] / np.linalg.norm(self.n,1);
            self.n[1] = self.n[1] / np.linalg.norm(self.n,1);
            self.n[2] = self.n[2] / np.linalg.norm(self.n,1); 
         
            fe_x[0] = self.loss_tensor[0].data.item() * np.exp(self.beta * (self.n[0] - np.max(self.n)));
            fe_x[1] = self.loss_tensor[1].data.item() * np.exp(self.beta * (self.n[1] - np.max(self.n)));
            fe_x[2] = self.loss_tensor[2].data.item() * np.exp(self.beta * (self.n[2] - np.max(self.n)));
            denom = fe_x[0] + fe_x[1] + fe_x[2]; 
            return (fe_x[self.i]/ denom)

                                               
        else :
            print("As of now, we only support 2 or 3 losses, please check input")

                            


    
    
    
    
      
    
 
