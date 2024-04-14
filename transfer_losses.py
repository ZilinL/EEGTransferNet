import torch
import torch.nn as nn
from loss_funcs import *
from loss_funcs import spd_mmd
from loss_funcs import riemann_lem
from loss_funcs import riemann_jeffrey
class TransferLoss(nn.Module):
    def __init__(self, loss_type, **kwargs):
        super(TransferLoss, self).__init__()
        self.loss_type = loss_type
        if loss_type == "mmd":
            self.loss_func = MMDLoss(**kwargs)
        elif loss_type == "spd_mmd":
            self.loss_func = spd_mmd.SPDMMDLoss(**kwargs)
        elif loss_type == "lmmd":
            self.loss_func = LMMDLoss(**kwargs)
        elif loss_type == "coral":
            self.loss_func = CORAL
        elif loss_type == "adv":
            self.loss_func = AdversarialLoss(**kwargs)
        elif loss_type == "daan":
            self.loss_func = DAANLoss(**kwargs)
        elif loss_type == "bnm":
            self.loss_func = BNM
        elif loss_type == "lem":
            self.loss_func = riemann_lem.LEM(**kwargs)            
        elif loss_type == "jeffrey":
            self.loss_func = riemann_jeffrey.Jeffrey(**kwargs)  
        else:
            print("WARNING: No valid transfer loss function is used.")
            self.loss_func = lambda x, y: 0 # return 0
    
    def forward(self, source, target, **kwargs):
        return self.loss_func(source, target, **kwargs)