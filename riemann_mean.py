import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import logm


class JefferyMean(nn.Module):
    def __init__(self):
        super(JefferyMean, self).__init__()

    def forward(self, X):  #X shape : torch.Size([36, 16, 1, 626])
        X = torch.squeeze(X)
        nbatch, nchannel, time = X.size()
        X_spd = torch.zeros(nbatch, nchannel, nchannel)
        E = torch.eye(nchannel).cuda()
        for i in range(nbatch):
            a = torch.cov(X[i,:,:])
            X_spd[i] = a + 0.001*torch.trace(a)*E
            # X_tmp[i] = torch.cov(X[i,:,:])+0.001*torch.mm(torch.trace(torch.cov(X[i,:,:])), torch.eye(nchannel))
        X_inv = torch.linalg.inv(X_spd)
        P = torch.mean(X_inv,0)
        Q = torch.mean(X_spd,0)

        # get P^(-0.5)
        L, V = torch.linalg.eig(P)
        L = torch.pow(L, -0.5)
        L = torch.diag(L)
        P_ = torch.mm(torch.mm(V, L), torch.linalg.inv(V)).float()   #V*L*V^(-1)

        # get P^(0.5)
        L, V = torch.linalg.eig(P)
        L = torch.pow(L, 0.5)
        L = torch.diag(L)
        # Q_ = torch.mm(torch.mm(V, L), torch.linalg.inv(V))
        P_P = torch.mm(torch.mm(V, L), torch.linalg.inv(V)).float()

        # get P_Q_P_^(0.5)
        # P_Q_P_ = torch.mm(torch.mm(P_, Q_), P_) 
        P_Q_P_ = torch.mm(torch.mm(P_P, Q), P_P) 
        L, V = torch.linalg.eig(P_Q_P_)
        L = torch.pow(L, 0.5)
        L = torch.diag(L)
        PQP_ = torch.mm(torch.mm(V, L), torch.linalg.inv(V)).float()

        X = torch.mm(torch.mm(P_, PQP_), torch.linalg.inv(P_)).float()

        return X


class LEMMean(nn.Module):
    def __init__(self):
        super(LEMMean, self).__init__()

    def forward(self, X):  #X shape : torch.Size([36, 16, 1, 626])
        X = torch.squeeze(X)
        nbatch, nchannel, time = X.size()
        X_spd = torch.zeros(nbatch, nchannel, nchannel)
        E = torch.eye(nchannel).cuda()
        for i in range(nbatch):
            a = torch.cov(X[i,:,:])
            X_spd[i] = a + 0.001*torch.trace(a)*E
            # X_tmp[i] = torch.cov(X[i,:,:])+0.001*torch.mm(torch.trace(torch.cov(X[i,:,:])), torch.eye(nchannel))

        # get log(x)
        L, V = torch.linalg.eig(X_spd)
        L = torch.log(L)
        L_log = torch.zeros(nbatch, nchannel, nchannel)
        # X_log = torch.zeros(nbatch, nchannel, nchannel)
        for i in range(nbatch):
            L_log[i] = torch.diag(L[i,:])
        X_log = torch.matmul(torch.matmul(V.float(),L_log),torch.linalg.inv(V).float())
            # X_log[i] = torch.mm(torch.mm(V[i,:,:].float(), L_log[i,:,:]), torch.linalg.inv(V)[i,:,:].float())
        X_mean = torch.mean(X_log,0)

        # get exp(x)
        L, V = torch.linalg.eig(X_mean)
        L = torch.exp(L)
        L = torch.diag(L)
        X = torch.mm(torch.mm(V, L), torch.linalg.inv(V))
        return X