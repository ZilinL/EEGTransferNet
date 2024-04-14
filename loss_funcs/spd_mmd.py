from numpy import ndarray
import torch
import torch.nn as nn
import pyriemann

class TangentVector(nn.Module):
    def __init__(self, estimator='cov'):
        super(TangentVector, self).__init__()
        self.estimator = estimator
    
    def torch2np(self, X):  #X shape : torch.Size([36, 16, 1, 626])
        X = torch.squeeze(X)
        np_X = X.cpu().detach().numpy()
        return np_X

    def forward(self, X):
        np_X = self.torch2np(X)
        cov_X = pyriemann.utils.covariance.covariances(np_X, self.estimator)  # X : ndarray, shape (n_matrices, n_channels, n_times)
        meanR_X = pyriemann.utils.mean.mean_riemann(cov_X, tol=1e-08, maxiter=50, init=None, sample_weight=None)
        tangent_np_X = pyriemann.utils.tangentspace.tangent_space(cov_X, meanR_X)
        tangent_X = torch.from_numpy(tangent_np_X)
        return tangent_X


class SPDMMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(SPDMMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type
        self.Tangent_Vector = TangentVector(estimator='cov')

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        source = self.Tangent_Vector(source)
        target = self.Tangent_Vector(target)
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss
