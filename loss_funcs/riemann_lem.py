import torch
import torch.nn as nn

class LEM(nn.Module):
    def __init__(self):
        super(LEM, self).__init__()
 
    def forward(self, source, target):  #X shape : torch.Size([16,16])
        # get log(x)
        L, V = torch.linalg.eig(source)
        L = torch.log(L)
        L = torch.diag(L)
        source_log = torch.mm(torch.mm(V, L), torch.linalg.inv(V))

        # get log(y)
        L, V = torch.linalg.eig(target)
        L = torch.log(L)
        L = torch.diag(L)
        target_log = torch.mm(torch.mm(V, L), torch.linalg.inv(V))
        
        X = torch.pow(torch.linalg.matrix_norm(torch.sub(source_log, target_log)),2)
        return X

