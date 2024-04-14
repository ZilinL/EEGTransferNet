import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import logm


class EuclideanMean(nn.Module):
    def __init__(self):
        super(EuclideanMean, self).__init__()

    def forward(self, X):  #X shape : torch.Size([36, 16, 1, 626])
        X = torch.squeeze(X)
        nbatch, nchannel, time = X.size()
        X_spd = torch.zeros(nbatch, nchannel, nchannel)
        E = torch.eye(nchannel).cuda()
        for i in range(nbatch):
            a = torch.cov(X[i,:,:])
            X_spd[i] = a + 0.001*torch.trace(a)*E
            # X_tmp[i] = torch.cov(X[i,:,:])+0.001*torch.mm(torch.trace(torch.cov(X[i,:,:])), torch.eye(nchannel))
        spd_mean = torch.mean(X_spd,0)

        return spd_mean

