############ FINITE DIFFERENCE #################

"""
loss Usage:
    
#    pass in 5 points as a np array 
#    outputs a forth order accurate first derivative approximation
#    if more accurate slope approximation is needed, then more points would be 
    # required
    
"""

class FiniteDifference():
    
    def __init__(self, loss_pts, args):
        
        self.loss_pts = loss_pts
        self.args = args
    
    
    def FD(self):
    
        # New technique:

        if self.args["fd_order"] == 5:
        
            der = ((25/12) * self.loss_pts[4]) - ((4) * self.loss_pts[3]) + ((3) * self.loss_pts[2]) \
        - ((4/3) * self.loss_pts[1]) + ((1/4) * self.loss_pts[0])
        
        elif self.args["fd_order"] == 3:
            der = (-3/2) * self.loss_pts[0] + 2 * self.loss_pts[1] + (-1/2) * self.loss_pts[2]
        
        else:
            raise NotImplementedError("A finite difference order of {} is not implemented yet.".format(self.args.fd_order))
    
    
        return der
 

  