import torch
import torch.nn as nn
import data_loader
from torchvision import models


def get_filter_head():
        return FilterBank()



# ========================================================================================        

#Depthwise separable convolution
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwiseconv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwiseconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding, dilation=dilation, groups=1, bias=bias)

    def forward(self, x):
        x = self.depthwiseconv(x)
        x = self.pointwiseconv(x)
        return x

# Depthwise convolution, need to understand.
class DepthwiseConv2D(nn.Conv2d):
    def __init__(self, *arg, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(DepthwiseConv2D, self).__init__(*arg, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(DepthwiseConv2D, self).forward(x)

# ======================================================================================

class FilterBank(nn.Module):
    def __init__(self, channels=22, kernel_length=125, kernel_length2=16, F1=8, F2=16, D=2, dropout_rate=0.5):
        super(FilterBank, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.channels = channels
        self.kernel_length = kernel_length
        self.kernel_length2 = kernel_length2
        self.dropout_rate = dropout_rate

        # Block1
        self.conv1 = nn.Conv2d(1, self.F1, (1,self.kernel_length), padding=0, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(self.F1 )
        self.depthwiseconv = DepthwiseConv2D(self.F1, self.F1*self.D, (self.channels,1), max_norm=1, groups=self.F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(self.F1*self.D)
        self.activate1 = nn.ELU()
        self.pooling1 = nn.AvgPool2d((1,4), stride=4)
        self.dropout1 = nn.Dropout(p=self.dropout_rate)

   
    def forward(self, x):

        x = self.conv1(x) # [1,1,59,300] -> [1,8,59,237]
        x = self.batchnorm1(x)
        filtered_data = self.depthwiseconv(x) # -> [1,16,1,237]
        x = self.batchnorm2(filtered_data)
        x = self.activate1(x)
        x = self.pooling1(x) # -> [1,16,1,59]
        x = self.dropout1(x)
    
        # output filtered data

        return x, filtered_data




