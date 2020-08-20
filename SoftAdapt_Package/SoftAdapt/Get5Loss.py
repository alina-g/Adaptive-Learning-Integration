
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
import numpy as np

class Get5Loss():
    
    def __init__(self, new_loss, index, args):
        self.new_loss = new_loss
        self.index = index
        self.args = args
    

    def get_5loss(self):
    
        if self.args["global_count"] > 4 :
        
            if self.index == 1:
    
                self.args["loss2"] = np.hstack( (self.args["loss2"] , self.new_loss) );
    
                if self.args["global_count"] >= self.args["fd_order"]:
                    self.args["loss2"] = self.args["loss2"][-self.args["fd_order"]:];
    
            elif self.index == 0:
        #             self.args.loss1[self.args.loss1_global_count % 5] = self.new_loss
    
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
            
            

        