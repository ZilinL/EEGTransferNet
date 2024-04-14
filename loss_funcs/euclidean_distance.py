import torch
import torch.nn as nn

class Fro(nn.Module):  # F2范数
    def __init__(self):
        super(Fro, self).__init__()
 
    def forward(self, source, target):  #X shape : torch.Size([16,16])
        d = torch.sub(source, target)
        X = torch.linalg.matrix_norm(d)

        return X

