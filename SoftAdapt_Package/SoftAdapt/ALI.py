import numpy as np



class Functions():
    
    def __init__(self, loss_pts, args, alpha, loss_num, new_loss, index, recent_loss):
        
        self.loss_pts = loss_pts
        self.args = args
        self.alpha = alpha
        self.loss_num = loss_num
        self.new_loss = new_loss
        self.index = index
        self.recent_loss = recent_loss
        
        
############ FINITE DIFFERENCE #################

    """
loss Usage:
    
#    pass in 5 points as a np array 
#    outputs a forth order accurate first derivative approximation
#    if more accurate slope approximation is needed, then more points would be 
    # required
    """
    
    
    def FD(self):
        
    
    # New technique:
    # New technique:
    
        if self.args["fd_order"] == 5:
            der = ((25/12) * self.loss_pts[4]) - ((4) * self.loss_pts[3]) + ((3) * self.loss_pts[2]) \
            - ((4/3) * self.loss_pts[1]) + ((1/4) * self.loss_pts[0])
            
        elif self.args["fd_order"] == 3:
            der = (-3/2) * self.loss_pts[0] + 2 * self.loss_pts[1] + (-1/2) * self.loss_pts[2]
        else:
            raise NotImplementedError("A finite difference order of {} is not implemented yet.".format(self.args.fd_order))
        
        
        return der
    
        
        
    
    
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
    
            
    """
    Get 5 Loss :
        it stores the last 5 losses efficently and properly 
        
        Usage : 
            
           inputs :  the most current loss value -> new_loss 
                     a string that indicates which part of the loss is being stored
                         strings should be of the format "loss1", "loss2", "loss3" etc
                     
                     
                     
                     a (preferably global ) dictionary with the loss vectors
                         -> args
                     
                     
            output :  it sets the global arrays loss1 and loss2 and 
            their corresponding counters
            
    """
    
    def get_5loss(self):
    
        if self.args["global_count"] > 4 :
        
            if self.index == 1:
    
                self.args["loss2"] = np.hstack( (self.args["loss2"] , self.new_loss) );
    
                if self.args["global_count"] >= self.args["fd_order"]:
                    self.args["loss2"] = self.args["loss2"][-self.args["fd_order"]:];
    
            elif self.index == 0:
        #             self.args.loss1[self.args.loss1_global_count % 5] = new_loss
    
                   # to save the arrays in an orderly fashion
                self.args["loss1"] = np.hstack((self.args["loss1"], self.new_loss));
                if self.args["global_count"] >= self.args["fd_order"]:
                    self.args["loss1"] = self.args["loss1"][-self.args["fd_order"]:];
    
    
            elif self.index == 2:
                self.args["loss3"] = np.hstack((self.args["loss3"], self.new_loss));
                 # to save the arrays in an orderly fashion
                self.args.loss3.append(self.new_loss);
    
                if self.args.loss3.size > 5 :
                      self.args["loss3"] = self.args["loss3"][-5:];
    
            else :
                 print("Wrong Index");
    
        
        else : 
            
            print("ALI idle --- less than 5 iters");
            self.args["global_count"] += 1;
     
    ###################### ADAPTIVE LEARNING INTEGRATION ##########################
    ################################# AdaLearn ####################################
    
    
    
    
    
    #################### avg_calc ############################
    """
    average calculator : given the most recent loss it will calculate the average with resepct
    to the average in a memory efficient way (I think)
    input : the most recent loss (for each individual loss)
    outputs : it changes the global variable self.args.loss1_avg (and for all other losses) and does
    not return a value
    """
    
    def avg_calc(self):
      
        if self.args["loss1_global_count"] > 4 :
            
            if self.args["loss1_avg"] == 0 : 
                
                self.args["loss1_avg"] = np.mean(self.args["loss2"]);
            
            self.args["loss1_avg"] = (self.args["loss1_avg"] * 5 + self.recent_loss)/6;
        
        
        
        
    
    
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
    
    def adapt_lr(self):
        
            
        if self.recent_loss > self.args["loss1_avg"] and self.args["flag"] == 'GO':
            self.args["lr"] = self.args["lr_max"];        
            self.args["flag"] = 'NO'
            self.args["adapt_iter2"] += 1;
    
    #         print("lr is inc to max");
            
        if self.recent_loss > self.args["loss1_avg"] and self.args["flag"] == 'NO' and  self.args["adapt_iter2"] > 2:
            
            self.args["lr"] = self.args["lr_min"];        
            self.args["flag"] = 'NO2'
            self.args["adapt_iter"] += 1;
    
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
    
            
    def lr_decay(self):
    
        if self.args["flag"] == 'NO2' and self.args["lr"] < self.args["lr_max"] : 
            print("OK AT LEAST IT GOES THROUGH IT!")
            self.args["lr"] *= 2;
            if self.args["lr"] > self.args["lr_max"] :
                
                    self.args["lr"] = self.args["lr_max"];
            
            
        elif self.args["flag"] == 'NO' and self.args["lr"] > self.args["lr_min"] : 
      
            if self.args["lr"] > self.args["lr_min"]: 
    
                self.args["lr"] = self.args["lr_max"] * np.exp(-1 * self.args["adapt_iter"] * self.args["kappa"]);
    
            if self.args["lr"] < self.args["lr_min"] : 
    
                self.args["lr"] = self.args["lr_min"];
    
            else :
    
                self.args["flag"] = 'GO'     
        else:
            
            self.args["flag"] = 'GO'
        
        

        
    # if the user does not want to have a seperate file, they can make all the-
    # needed parameters here with this function
                     
                
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
        
        
        