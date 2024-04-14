import torch
import torch.nn as nn
import numpy as np
import pyriemann as pyr
from scipy.linalg import logm


class CenterAlignment(nn.Module):
    def __init__(self, CA_type):
        super(CenterAlignment, self).__init__()
        self.CA_type = CA_type

    def forward(self, X):  #X shape : torch.Size([36, 16, 1, 626])
        nbatch, nchannel, _, time = X.size()
        X_cov = pyr.utils.covariance.covariances(torch.squeeze(X).detach().cpu())
        CA_mean = torch.from_numpy(pyr.utils.mean.mean_covariance(X_cov, metric=self.CA_type))
        
        # get CA_mean^(-0.5)
        L, V = torch.linalg.eig(CA_mean)
        L = torch.pow(L, -0.5)
        L = torch.diag(L)
        CA_M = torch.mm(torch.mm(V, L), torch.linalg.inv(V)).float().cuda()
        CA_M_repeat = CA_M.repeat(nbatch,1,1)
        X = torch.squeeze(X)
        X_CA =  torch.bmm(CA_M_repeat, X)
        X_CA = torch.unsqueeze(X_CA,2)
        return X_CA

