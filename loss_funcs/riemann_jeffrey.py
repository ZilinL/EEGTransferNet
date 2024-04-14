import torch
import torch.nn as nn

class Jeffrey(nn.Module):
    def __init__(self):
        super(Jeffrey, self).__init__()
 
    def forward(self, source, target):  #X shape : torch.Size([16,16])
        X_inv = torch.linalg.inv(source)
        Y_inv = torch.linalg.inv(target)
        X_inv_Y = torch.trace(torch.mm(X_inv, target))
        Y_inv_X = torch.trace(torch.mm(Y_inv, source))
        X = 0.5*X_inv_Y + 0.5*Y_inv_X - source.size()[0]
        X = torch.abs(X)
        return X

