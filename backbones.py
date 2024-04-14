import torch
import torch.nn as nn
import data_loader
from torchvision import models

resnet_dict = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
}

def get_filter_head(name):
        # return FilterBank() # FilterBank() MyFilterBank() DeepConveNet_Filter() ShallowConveNet_Filter()
    if "filter2" in name.lower():
        return FilterBank()
    elif "filter1" == name.lower():
        return FilterBank1()
    elif "filter3" == name.lower():
        return FilterBank3()



def get_backbone(name):
    if "resnet" in name.lower():
        return ResNetBackbone(name)
    elif "alexnet" == name.lower():
        return AlexNetBackbone()
    elif "dann" == name.lower():
        return DaNNBackbone()
    elif "eegnet" == name.lower():
        return EEGNetBackbone()
    elif "eegtransfernet" == name.lower():
        return EEGTransferNetBackbone()
    elif "eegtransfernet1" == name.lower():
        return EEGTransferNetBackbone1()
    elif "eegtransfernet3" == name.lower():
        return EEGTransferNetBackbone3()    
    elif "myeegnet" == name.lower():
        return MyEEGBackbone()
    elif "deepconvnetbody" == name.lower():
        return DeepConveNet_Body()
    elif "deepconvnet" == name.lower():
        return DeepConveNet()
    elif "shallowconvenet" == name.lower():
        return ShallowConveNet()        
    elif "shallowconvenetbody" == name.lower():
        return ShallowConveNet_Body()  

# ========================================================================================        

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
        self.max_norm = max_norm


    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)
    
    
class LazyLinearWithConstraint(nn.LazyLinear):
    def __init__(self, *args, max_norm=1., **kwargs):
        super(LazyLinearWithConstraint, self).__init__(*args, **kwargs)
        self.max_norm = max_norm
        
    
    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return self(x)

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

class EEGNetBackbone(nn.Module):
    def __init__(self, channels=56, kernel_length=50, kernel_length2=16, F1=8, F2=16, D=2, dropout_rate=0.5): #ERN: kernel_length=50, kernel_length2=16, F1=8, F2=16, D=2, dropout_rate=0.5) (RSVP: kernel_length=16, kernel_length2=4)(MI:kernel_length=125, kernel_length2=64, F1=8, F2=16, D=2, dropout_rate=0.5)
        super(EEGNetBackbone, self).__init__()
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
        self.pooling1 = nn.AvgPool2d((1,4), stride=4) #MI和ERN都采用4
        self.dropout1 = nn.Dropout(p=self.dropout_rate)

        #Block2
        self.separableconv = SeparableConv2d(self.F1*self.D, self.F2, (1, self.kernel_length2))
        self.batchnorm3 = nn.BatchNorm2d(self.F2)
        self.activate2 = nn.ELU()
        self.pooling2 = nn.AvgPool2d((1,8), stride=8) #MI和ERN都采用8
        self.dropout2 = nn.Dropout(p=self.dropout_rate)
   
    def forward(self, x):

        x = self.conv1(x) # [1,1,59,300] -> [1,8,59,237]
        x = self.batchnorm1(x)
        x = self.depthwiseconv(x) # -> [1,16,1,237]
        x = self.batchnorm2(x)
        x = self.activate1(x)
        x = self.pooling1(x) # -> [1,16,1,59]
        x = self.dropout1(x)
        
        x = self.separableconv(x) # -> [1,16,1,44]
        x = self.batchnorm3(x)
        x = self.activate2(x)
        x = self.pooling2(x) # -> [1,16,1,5]
        x = self.dropout2(x)
        x = x.view(x.size(0),-1)
        # output feature

        return x

    def output_num(self, **kwargs):
        folder_scr = kwargs['folder_src']
        batch_size = kwargs['batch_size']
        num_workers = kwargs['num_workers']
        _, _, num_channels, trial_length,_ = data_loader.load_data(folder_scr, batch_size, True, num_workers)
        data_tmp = torch.rand(1, 1, num_channels, trial_length)
        EEGNetBackbone_tmp = EEGNetBackbone(num_channels, kernel_length=self.kernel_length, kernel_length2=self.kernel_length2, F1=self.F1, F2=self.F2, D=self.D, dropout_rate=0.5)
        EEGNetBackbone_tmp.eval()
        _feature_dim = EEGNetBackbone_tmp(data_tmp).view(-1,1).shape[0]
        return _feature_dim

########################################################################################################################################################

class FilterBank(nn.Module):  # ERN: kernel_length=50, kernel_length2=16, F1=8, F2=16, D=2, dropout_rate=0.5) (RSVP: kernel_length=16, kernel_length2=4)(MI:kernel_length=125, kernel_length2=64, F1=8, F2=16, D=2, dropout_rate=0.5)
    def __init__(self, channels=56, kernel_length=50, F1=8, F2=16, D=2, dropout_rate=0.5):  #OpenBMI: channels=20, kernel_length=125, F1=8, F2=16, D=2, dropout_rate=0.5
        super(FilterBank, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.channels = channels
        self.kernel_length = kernel_length
        self.dropout_rate = dropout_rate

        # Block1
        self.conv1 = nn.Conv2d(1, self.F1, (1,self.kernel_length), padding=0, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(self.F1 )
        self.depthwiseconv = DepthwiseConv2D(self.F1, self.F1*self.D, (self.channels,1), max_norm=1, groups=self.F1, bias=False)
        # self.batchnorm2 = nn.BatchNorm2d(self.F1*self.D)
        # self.activate1 = nn.ELU()
        # self.pooling1 = nn.AvgPool2d((1,4), stride=4)
        # self.dropout1 = nn.Dropout(p=self.dropout_rate)

   
    def forward(self, x):

        x = self.conv1(x) # [1,1,59,300] -> [1,8,59,237]
        x = self.batchnorm1(x)
        x = self.depthwiseconv(x) # -> [1,16,1,237]
        # x = self.batchnorm2(x)
        # x = self.activate1(x)
        # x = self.pooling1(x) # -> [1,16,1,59]
        # x = self.dropout1(x)
    
        # output filtered data

        return x
    
class EEGTransferNetBackbone(nn.Module):
    def __init__(self, channels=56, kernel_length=50, kernel_length2=16, F1=8, F2=16, D=2, dropout_rate=0.5):
        super(EEGTransferNetBackbone, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.channels = channels
        self.kernel_length = kernel_length
        self.kernel_length2 = kernel_length2
        self.dropout_rate = dropout_rate

        # Block1
        # self.conv1 = nn.Conv2d(1, self.F1, (1,self.kernel_length), padding=0, bias=False)
        # self.batchnorm1 = nn.BatchNorm2d(self.F1 )
        # self.depthwiseconv = DepthwiseConv2D(self.F1, self.F1*self.D, (self.channels,1), max_norm=1, groups=self.F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(self.F1*self.D)
        self.activate1 = nn.ELU()
        self.pooling1 = nn.AvgPool2d((1,4), stride=4)
        self.dropout1 = nn.Dropout(p=self.dropout_rate)

        #Block2
        self.separableconv = SeparableConv2d(self.F1*self.D, self.F2, (1, self.kernel_length2))
        self.batchnorm3 = nn.BatchNorm2d(self.F2)
        self.activate2 = nn.ELU()
        self.pooling2 = nn.AvgPool2d((1,8), stride=8)
        self.dropout2 = nn.Dropout(p=self.dropout_rate)
   
    def forward(self, x):

        x = self.batchnorm2(x)
        x = self.activate1(x)
        x = self.pooling1(x) # -> [1,16,1,59]
        x = self.dropout1(x)
        
        x = self.separableconv(x) # -> [1,16,1,44]
        x = self.batchnorm3(x)
        x = self.activate2(x)
        x = self.pooling2(x) # -> [1,16,1,5]
        x = self.dropout2(x)
        x = x.view(x.size(0),-1)
        # output feature

        return x

    def output_num(self, **kwargs):
        folder_scr = kwargs['folder_src']
        batch_size = kwargs['batch_size']
        num_workers = kwargs['num_workers']
        _, _, num_channels, trial_length, _ = data_loader.load_data(folder_scr, batch_size, True, num_workers)
        data_tmp = torch.rand(1, 1, num_channels, trial_length)
        EEGNetBackbone_tmp = EEGNetBackbone(num_channels, kernel_length=self.kernel_length, kernel_length2=self.kernel_length2, F1=self.F1, F2=self.F2, D=self.D, dropout_rate=0.5)
        EEGNetBackbone_tmp.eval()
        _feature_dim = EEGNetBackbone_tmp(data_tmp).view(-1,1).shape[0]
        return _feature_dim

#############################################################################################################################
class MyEEGBackbone(nn.Module):
    def __init__(self, channels=22, kernel_length=125, kernel_length2=64, F1=16, F2=16, D=1, dropout_rate=0.5):
        super(MyEEGBackbone, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.channels = channels
        self.kernel_length = kernel_length
        self.kernel_length2 = kernel_length2
        self.dropout_rate = dropout_rate

        # Block1
        self.conv1 = nn.Conv2d(1, self.F1, (self.channels,1), padding=0, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(self.F1 )
        self.activate1 = nn.ELU()
        self.pooling1 = nn.AvgPool2d((1,4), stride=4) #MI和ERN都采用4
        self.dropout1 = nn.Dropout(p=self.dropout_rate)

        #Block2
        self.separableconv = SeparableConv2d(self.F1*self.D, self.F2, (1, self.kernel_length2))
        self.batchnorm3 = nn.BatchNorm2d(self.F2)
        self.activate2 = nn.ELU()
        self.pooling2 = nn.AvgPool2d((1,8), stride=8) #MI和ERN都采用8
        self.dropout2 = nn.Dropout(p=self.dropout_rate)
   
    def forward(self, x):

        x = self.conv1(x) # [1,1,59,300] -> [1,8,59,237]
        x = self.batchnorm1(x)
        x = self.activate1(x)
        x = self.pooling1(x) # -> [1,16,1,59]
        x = self.dropout1(x)
        
        x = self.separableconv(x) # -> [1,16,1,44]
        x = self.batchnorm3(x)
        x = self.activate2(x)
        x = self.pooling2(x) # -> [1,16,1,5]
        x = self.dropout2(x)
        x = x.view(x.size(0),-1)
        # output feature

        return x

    def output_num(self, **kwargs):
        folder_scr = kwargs['folder_src']
        batch_size = kwargs['batch_size']
        num_workers = kwargs['num_workers']
        _, _, num_channels, trial_length,_ = data_loader.load_data(folder_scr, batch_size, True, num_workers)
        data_tmp = torch.rand(1, 1, num_channels, trial_length)
        EEGNetBackbone_tmp = MyEEGBackbone(num_channels, kernel_length=self.kernel_length, kernel_length2=self.kernel_length2, F1=self.F1, F2=self.F2, D=self.D, dropout_rate=0.5)
        EEGNetBackbone_tmp.eval()
        _feature_dim = EEGNetBackbone_tmp(data_tmp).view(-1,1).shape[0]
        return _feature_dim
    
########################################################################################################################################################################################
class FilterBank1(nn.Module):
    def __init__(self, channels=22, kernel_length=125, F1=8, F2=16, D=2, dropout_rate=0.5):
        super(FilterBank1, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.channels = channels
        self.kernel_length = kernel_length
        self.dropout_rate = dropout_rate

        # Block1
        self.conv1 = nn.Conv2d(1, self.F1, (1,self.kernel_length), padding=0, bias=False)
        # self.batchnorm1 = nn.BatchNorm2d(self.F1 )
        # self.batchnorm2 = nn.BatchNorm2d(self.F1*self.D)
        # self.activate1 = nn.ELU()
        # self.pooling1 = nn.AvgPool2d((1,4), stride=4)
        # self.dropout1 = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):

        x = self.conv1(x) # [1,1,59,300] -> [1,8,59,237]
        # x = self.batchnorm1(x)
        # x = self.depthwiseconv(x) # -> [1,16,1,237]
        # x = self.batchnorm2(x)
        # x = self.activate1(x)
        # x = self.pooling1(x) # -> [1,16,1,59]
        # x = self.dropout1(x)
    
        # output filtered data

        return x

class EEGTransferNetBackbone1(nn.Module):
    def __init__(self, channels=22, kernel_length=125, kernel_length2=64, F1=8, F2=16, D=2, dropout_rate=0.5):
        super(EEGTransferNetBackbone1, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.channels = channels
        self.kernel_length = kernel_length
        self.kernel_length2 = kernel_length2
        self.dropout_rate = dropout_rate

        # Block1
        self.batchnorm1 = nn.BatchNorm2d(self.F1 )
        self.depthwiseconv = DepthwiseConv2D(self.F1, self.F1*self.D, (self.channels,1), max_norm=1, groups=self.F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(self.F1*self.D)
        self.activate1 = nn.ELU()
        self.pooling1 = nn.AvgPool2d((1,4), stride=4)
        self.dropout1 = nn.Dropout(p=self.dropout_rate)

        #Block2
        self.separableconv = SeparableConv2d(self.F1*self.D, self.F2, (1, self.kernel_length2))
        self.batchnorm3 = nn.BatchNorm2d(self.F2)
        self.activate2 = nn.ELU()
        self.pooling2 = nn.AvgPool2d((1,8), stride=8)
        self.dropout2 = nn.Dropout(p=self.dropout_rate)
   
    def forward(self, x):
        x = self.batchnorm1(x)
        x = self.depthwiseconv(x)
        x = self.batchnorm2(x)
        x = self.activate1(x)
        x = self.pooling1(x) # -> [1,16,1,59]
        x = self.dropout1(x)
        
        x = self.separableconv(x) # -> [1,16,1,44]
        x = self.batchnorm3(x)
        x = self.activate2(x)
        x = self.pooling2(x) # -> [1,16,1,5]
        x = self.dropout2(x)
        x = x.view(x.size(0),-1)
        # output feature

        return x

    def output_num(self, **kwargs):
        folder_scr = kwargs['folder_src']
        batch_size = kwargs['batch_size']
        num_workers = kwargs['num_workers']
        _, _, num_channels, trial_length,_ = data_loader.load_data(folder_scr, batch_size, True, num_workers)
        data_tmp = torch.rand(1, 1, num_channels, trial_length)
        EEGNetBackbone_tmp = EEGNetBackbone(num_channels, kernel_length=self.kernel_length, kernel_length2=self.kernel_length2, F1=self.F1, F2=self.F2, D=self.D, dropout_rate=0.5)
        EEGNetBackbone_tmp.eval()
        _feature_dim = EEGNetBackbone_tmp(data_tmp).view(-1,1).shape[0]
        return _feature_dim
    
###############################################放在第三个卷积层后##############################################################    
class FilterBank3(nn.Module):
    def __init__(self, channels=8, kernel_length=16, kernel_length2=4, F1=8, F2=16, D=2, dropout_rate=0.5):
        super(FilterBank3, self).__init__()
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
        self.pooling1 = nn.AvgPool2d((1,2), stride=2)
        self.dropout1 = nn.Dropout(p=self.dropout_rate)

        #Block2
        self.separableconv = SeparableConv2d(self.F1*self.D, self.F2, (1, self.kernel_length2))

    def forward(self, x):

        x = self.conv1(x) # [1,1,59,300] -> [1,8,59,237]
        x = self.batchnorm1(x)
        x = self.depthwiseconv(x) # -> [1,16,1,237]
        x = self.batchnorm2(x)
        x = self.activate1(x)
        x = self.pooling1(x) # -> [1,16,1,59]
        x = self.dropout1(x)

        x = self.separableconv(x) # -> [1,16,1,44]           

        # output filtered data

        return x

class EEGTransferNetBackbone3(nn.Module):
    def __init__(self, channels=8, kernel_length=16, kernel_length2=4, F1=8, F2=16, D=2, dropout_rate=0.5):
        super(EEGTransferNetBackbone3, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.channels = channels
        self.kernel_length = kernel_length
        self.kernel_length2 = kernel_length2
        self.dropout_rate = dropout_rate

        # Block1
        # self.batchnorm1 = nn.BatchNorm2d(self.F1 )
        # self.depthwiseconv = DepthwiseConv2D(self.F1, self.F1*self.D, (self.channels,1), max_norm=1, groups=self.F1, bias=False)
        # self.batchnorm2 = nn.BatchNorm2d(self.F1*self.D)
        # self.activate1 = nn.ELU()
        # self.pooling1 = nn.AvgPool2d((1,4), stride=4)
        # self.dropout1 = nn.Dropout(p=self.dropout_rate)

        #Block2
        # self.separableconv = SeparableConv2d(self.F1*self.D, self.F2, (1, self.kernel_length2))
        self.batchnorm3 = nn.BatchNorm2d(self.F2)
        self.activate2 = nn.ELU()
        self.pooling2 = nn.AvgPool2d((1,4), stride=4)
        self.dropout2 = nn.Dropout(p=self.dropout_rate)
   
    def forward(self, x):
        # x = self.batchnorm1(x)
        # x = self.depthwiseconv(x)
        # x = self.batchnorm2(x)
        # x = self.activate1(x)
        # x = self.pooling1(x) # -> [1,16,1,59]
        # x = self.dropout1(x)
        
        # x = self.separableconv(x) # -> [1,16,1,44]
        x = self.batchnorm3(x)
        x = self.activate2(x)
        x = self.pooling2(x) # -> [1,16,1,5]
        x = self.dropout2(x)
        x = x.view(x.size(0),-1)
        # output feature

        return x

    def output_num(self, **kwargs):
        folder_scr = kwargs['folder_src']
        batch_size = kwargs['batch_size']
        num_workers = kwargs['num_workers']
        _, _, num_channels, trial_length,_ = data_loader.load_data(folder_scr, batch_size, True, num_workers)
        data_tmp = torch.rand(1, 1, num_channels, trial_length)
        EEGNetBackbone_tmp = EEGNetBackbone(num_channels, kernel_length=self.kernel_length, kernel_length2=self.kernel_length2, F1=self.F1, F2=self.F2, D=self.D, dropout_rate=0.5)
        EEGNetBackbone_tmp.eval()
        _feature_dim = EEGNetBackbone_tmp(data_tmp).view(-1,1).shape[0]
        return _feature_dim


###################################################################################################
class DeepConveNet(nn.Module):
    def __init__(self, channels=20, kernel_length=125, kernel_length2=64, F1=8, F2=16, D=2, dropout_rate=0.5):
        super(DeepConveNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.channels = channels
        self.kernel_length = kernel_length
        self.kernel_length2 = kernel_length2
        self.dropout_rate = dropout_rate

        # Block1
        self.conv0 = Conv2dWithConstraint(1, 25, (1,10), max_norm=2, padding=0, bias=False)
        self.conv1 = Conv2dWithConstraint(25, 25, (self.channels, 1), max_norm=2, padding=0, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(25)
        self.activate1 = nn.ELU()
        self.maxpool1 = nn.MaxPool2d((1,3), stride=(1,3))
        self.dropout1 = nn.Dropout(p=0.5)

        #Block2
        self.conv2 = Conv2dWithConstraint(25, 50, (1,10), max_norm=2, padding=0, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(50)
        self.activate2 = nn.ELU()
        self.maxpool2 = nn.MaxPool2d((1,3), stride=(1,3))
        self.dropout2 = nn.Dropout(p=0.5)

        #Block3
        self.conv3 = Conv2dWithConstraint(50, 100, (1,10), max_norm=2, padding=0, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(100)
        self.activate3 = nn.ELU()
        self.maxpool3 = nn.MaxPool2d((1,3), stride=(1,3))
        self.dropout3 = nn.Dropout(p=0.5)

        #Block4
        self.conv4 = Conv2dWithConstraint(100, 200, (1,10), max_norm=2, padding=0, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(200)
        self.activate4 = nn.ELU()
        self.maxpool4 = nn.MaxPool2d((1,3), stride=(1,3))
        self.dropout4 = nn.Dropout(p=0.5)
   
    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x) 
        x = self.batchnorm1(x)
        x = self.activate1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x) 
        x = self.batchnorm2(x)
        x = self.activate2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = self.conv3(x) 
        x = self.batchnorm3(x)
        x = self.activate3(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)

        x = self.conv4(x) 
        x = self.batchnorm4(x)
        x = self.activate4(x)
        x = self.maxpool4(x)
        x = self.dropout4(x)

        x = x.view(x.size(0),-1)
        # output feature

        return x

    def output_num(self, **kwargs):
        folder_scr = kwargs['folder_src']
        batch_size = kwargs['batch_size']
        num_workers = kwargs['num_workers']
        _, _, num_channels, trial_length,_ = data_loader.load_data(folder_scr, batch_size, True, num_workers)
        data_tmp = torch.rand(1, 1, num_channels, trial_length)
        EEGNetBackbone_tmp = DeepConveNet(num_channels, kernel_length=self.kernel_length, kernel_length2=self.kernel_length2, F1=self.F1, F2=self.F2, D=self.D, dropout_rate=0.5)
        EEGNetBackbone_tmp.eval()
        _feature_dim = EEGNetBackbone_tmp(data_tmp).view(-1,1).shape[0]
        return _feature_dim

class DeepConveNet_Filter(nn.Module):
    def __init__(self, channels=22, kernel_length=50, kernel_length2=16, F1=8, F2=16, D=2, dropout_rate=0.5):
        super(DeepConveNet_Filter, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.channels = channels
        self.kernel_length = kernel_length
        self.kernel_length2 = kernel_length2
        self.dropout_rate = dropout_rate

        # Block1
        self.conv0 = Conv2dWithConstraint(1, 25, (1,10), max_norm=2, padding=0, bias=False)
        self.conv1 = Conv2dWithConstraint(25, 25, (self.channels, 1), max_norm=2, padding=0, bias=False)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x) 
        return x

class DeepConveNet_Body(nn.Module):       
    def __init__(self, channels=8, kernel_length=16, kernel_length2=4, F1=8, F2=16, D=2, dropout_rate=0.5):
        super(DeepConveNet_Body, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.channels = channels
        self.kernel_length = kernel_length
        self.kernel_length2 = kernel_length2
        self.dropout_rate = dropout_rate


        self.batchnorm1 = nn.BatchNorm2d(25)
        self.activate1 = nn.ELU()
        self.maxpool1 = nn.MaxPool2d((1,3), stride=(1,3))
        self.dropout1 = nn.Dropout(p=0.5)

        #Block2
        self.conv2 = Conv2dWithConstraint(25, 50, (1,10), max_norm=2, padding=0, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(50)
        self.activate2 = nn.ELU()
        self.maxpool2 = nn.MaxPool2d((1,3), stride=(1,3))
        self.dropout2 = nn.Dropout(p=0.5)

        #Block3
        self.conv3 = Conv2dWithConstraint(50, 100, (1,10), max_norm=2, padding=0, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(100)
        self.activate3 = nn.ELU()
        self.maxpool3 = nn.MaxPool2d((1,3), stride=(1,3))
        self.dropout3 = nn.Dropout(p=0.5)

        #Block4
        self.conv4 = Conv2dWithConstraint(100, 200, (1,10), max_norm=2, padding=0, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(200)
        self.activate4 = nn.ELU()
        self.maxpool4 = nn.MaxPool2d((1,3), stride=(1,3))
        self.dropout4 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.batchnorm1(x)
        x = self.activate1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x) 
        x = self.batchnorm2(x)
        x = self.activate2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = self.conv3(x) 
        x = self.batchnorm3(x)
        x = self.activate3(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)

        x = self.conv4(x) 
        x = self.batchnorm4(x)
        x = self.activate4(x)
        x = self.maxpool4(x)
        x = self.dropout4(x)

        x = x.view(x.size(0),-1)
        # output feature

        return x

    def output_num(self, **kwargs):
        folder_scr = kwargs['folder_src']
        batch_size = kwargs['batch_size']
        num_workers = kwargs['num_workers']
        _, _, num_channels, trial_length,_ = data_loader.load_data(folder_scr, batch_size, True, num_workers)
        data_tmp = torch.rand(1, 1, num_channels, trial_length)
        EEGNetBackbone_tmp = DeepConveNet(num_channels, kernel_length=self.kernel_length, kernel_length2=self.kernel_length2, F1=self.F1, F2=self.F2, D=self.D, dropout_rate=0.5)
        EEGNetBackbone_tmp.eval()
        _feature_dim = EEGNetBackbone_tmp(data_tmp).view(-1,1).shape[0]
        return _feature_dim

#===========================================================================================
#========================================ShallowConvNet=====================================
class ShallowConveNet(nn.Module):
    def __init__(self, channels=22, kernel_length=125, kernel_length2=64, F1=8, F2=16, D=2, dropout_rate=0.5):
        super(ShallowConveNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.channels = channels
        self.kernel_length = kernel_length
        self.kernel_length2 = kernel_length2
        self.dropout_rate = dropout_rate

        # Block1
        self.conv0 = nn.Conv2d(1, 40, (1,25), padding=0, bias=False)
        self.conv1 = nn.Conv2d(40, 40, (self.channels, 1), padding=0, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(40)
        self.pooling1 = nn.AvgPool2d((1,75), stride=(1,15))
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x) 
        x = self.batchnorm1(x)
        x = torch.square(x)
        x = self.pooling1(x)
        x = torch.log(x)

        x = self.dropout1(x)
        x = x.view(x.size(0),-1)
        # output feature

        return x

    def output_num(self, **kwargs):
        folder_scr = kwargs['folder_src']
        batch_size = kwargs['batch_size']
        num_workers = kwargs['num_workers']
        _, _, num_channels, trial_length,_ = data_loader.load_data(folder_scr, batch_size, True, num_workers)
        data_tmp = torch.rand(1, 1, num_channels, trial_length)
        EEGNetBackbone_tmp = ShallowConveNet(num_channels, kernel_length=self.kernel_length, kernel_length2=self.kernel_length2, F1=self.F1, F2=self.F2, D=self.D, dropout_rate=0.5)
        EEGNetBackbone_tmp.eval()
        _feature_dim = EEGNetBackbone_tmp(data_tmp).view(-1,1).shape[0]
        return _feature_dim

class ShallowConveNet_Filter(nn.Module):
    def __init__(self, channels=22, kernel_length=50, kernel_length2=16, F1=8, F2=16, D=2, dropout_rate=0.5):
        super(ShallowConveNet_Filter, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.channels = channels
        self.kernel_length = kernel_length
        self.kernel_length2 = kernel_length2
        self.dropout_rate = dropout_rate

        # Block1
        self.conv0 = nn.Conv2d(1, 40, (1,25), padding=0, bias=False)
        self.conv1 = nn.Conv2d(40, 40, (self.channels, 1), padding=0, bias=False)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x) 
        return x

class ShallowConveNet_Body(nn.Module):
    def __init__(self, channels=22, kernel_length=50, kernel_length2=16, F1=8, F2=16, D=2, dropout_rate=0.5):
        super(ShallowConveNet_Body, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.channels = channels
        self.kernel_length = kernel_length
        self.kernel_length2 = kernel_length2
        self.dropout_rate = dropout_rate

        # Block1
        self.batchnorm1 = nn.BatchNorm2d(40)
        self.pooling1 = nn.AvgPool2d((1,75), stride=(1,15))
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.batchnorm1(x)
        x = torch.square(x)
        x = self.pooling1(x)
        x = torch.log(x)

        x = self.dropout1(x)
        x = x.view(x.size(0),-1)
        # output feature

        return x

    def output_num(self, **kwargs):
        folder_scr = kwargs['folder_src']
        batch_size = kwargs['batch_size']
        num_workers = kwargs['num_workers']
        _, _, num_channels, trial_length,_ = data_loader.load_data(folder_scr, batch_size, True, num_workers)
        data_tmp = torch.rand(1, 1, num_channels, trial_length)
        EEGNetBackbone_tmp = ShallowConveNet(num_channels, kernel_length=self.kernel_length, kernel_length2=self.kernel_length2, F1=self.F1, F2=self.F2, D=self.D, dropout_rate=0.5)
        EEGNetBackbone_tmp.eval()
        _feature_dim = EEGNetBackbone_tmp(data_tmp).view(-1,1).shape[0]
        return _feature_dim


############################################################################################
##########################        FBCNet       ###########################################
#############################################################################################
## Support classes for FBNet Implementation
class VarLayer(nn.Module):
    '''
    The variance layer: calculates the variance of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(VarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.var(dim = self.dim, keepdim= True)

class StdLayer(nn.Module):
    '''
    The standard deviation layer: calculates the std of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(StdLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.std(dim = self.dim, keepdim=True)

class LogVarLayer(nn.Module):
    '''
    The log variance layer: calculates the log variance of the data along given 'dim'
    (natural logarithm)
    '''
    def __init__(self, dim):
        super(LogVarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(x.var(dim = self.dim, keepdim= True), 1e-6, 1e6))

class MeanLayer(nn.Module):
    '''
    The mean layer: calculates the mean of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(MeanLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim = self.dim, keepdim=True)

class MaxLayer(nn.Module):
    '''
    The max layer: calculates the max of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(MaxLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        ma ,ima = x.max(dim = self.dim, keepdim=True)
        return ma

class swish(nn.Module):
    '''
    The swish layer: implements the swish activation function
    '''
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)

class FBCNet(nn.Module):
    # just a FBCSP like structure : chan conv and then variance along the time axis
    '''
        FBNet with seperate variance for every 1s. 
        The data input is in a form of batch x 1 x chan x time x filterBand
    '''
    def SCB(self, m, nChan, nBands, doWeightNorm=True, *args, **kwargs):
        '''
        The spatial convolution block
        m : number of sptatial filters.
        nBands: number of bands in the data
        '''
        return nn.Sequential(
                Conv2dWithConstraint(nBands, m*nBands, (nChan, 1), groups= nBands,
                                     max_norm = 2 , doWeightNorm = doWeightNorm,padding = 0),
                nn.BatchNorm2d(m*nBands),
                swish()
                )

    def LastBlock(self, inF, outF, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
                LinearWithConstraint(inF, outF, max_norm = 0.5, doWeightNorm = doWeightNorm, *args, **kwargs),
                nn.LogSoftmax(dim = 1))

    def __init__(self, nChan, nTime, nClass = 2, nBands = 9, m = 32,
                 temporalLayer = 'LogVarLayer', strideFactor= 4, doWeightNorm = True, *args, **kwargs):
        super(FBCNet, self).__init__()

        self.nBands = nBands
        self.m = m
        self.strideFactor = strideFactor

        # create all the parrallel SCBc
        self.scb = self.SCB(m, nChan, self.nBands, doWeightNorm = doWeightNorm)
        
        # Formulate the temporal agreegator
        # self.temporalLayer = current_module.__dict__[temporalLayer](dim = 3)

        # The final fully connected layer
        self.lastLayer = self.LastBlock(self.m*self.nBands*self.strideFactor, nClass, doWeightNorm = doWeightNorm)

    def forward(self, x):
        x = torch.squeeze(x.permute((0,4,2,3,1)), dim = 4)
        x = self.scb(x)
        x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3]/self.strideFactor)])
        x = self.temporalLayer(x)
        x = torch.flatten(x, start_dim= 1)
        x = self.lastLayer(x)
        return x
############################################################################################
##########################        finetune       ###########################################
############################################################################################
class DaNNBackbone(nn.Module):
    def __init__(self, n_input=224*224*3, n_hidden=256):
        super(DaNNBackbone, self).__init__()
        self.layer_input = nn.Linear(n_input, n_hidden)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self._feature_dim = n_hidden

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x

    def output_num(self):
        return self._feature_dim
    
# convnet without the last layer
class AlexNetBackbone(nn.Module):
    def __init__(self):
        super(AlexNetBackbone, self).__init__()
        model_alexnet = models.alexnet(pretrained=True)
        self.features = model_alexnet.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module(
                "classifier"+str(i), model_alexnet.classifier[i])
        self._feature_dim = model_alexnet.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self._feature_dim

class ResNetBackbone(nn.Module):
    def __init__(self, network_type):
        super(ResNetBackbone, self).__init__()
        resnet = resnet_dict[network_type](pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self._feature_dim = resnet.fc.in_features
        del resnet
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def output_num(self):
        return self._feature_dim

############################################################################################
##########################        EEG-Inception       ######################################
############################################################################################

class CustomPad(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        return nn.functional.pad(x, self.padding)


class DepthWiseConv2d(nn.Module):
    def __init__(self, channel, kernel_size, depth_multiplier, bias=False):
        super().__init__()
        self.nets = nn.ModuleList([
            nn.Conv2d(channel, channel, kernel_size, bias=bias, groups=channel)
            for _ in range(depth_multiplier)
        ])

    def forward(self, x):
        output = torch.cat([net(x) for net in self.nets], 1)
        return output


class EEGInception(nn.Module):
    def __init__(self, num_classes, fs=1282, num_channels=8, filters_per_branch=8,
                 scales_time=(500, 250, 125), dropout_rate=0.25,
                 activation=nn.ELU(inplace=True)):
        super().__init__()
        scales_samples = [int(s * fs / 1000) for s in scales_time]
        # ========================== BLOCK 1: INCEPTION ========================== #
        self.inception1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    1, filters_per_branch, (scales_sample, 1),
                    padding="same"
                    # padding=((scales_sample - 1) // 2, 0)
                ) if torch.__version__ >= "1.9" else nn.Sequential(
                    CustomPad((0, 0,scales_sample // 2 - 1, scales_sample // 2, )),
                    nn.Conv2d(
                        1, filters_per_branch, (scales_sample, 1)
                    )
                ),
                nn.BatchNorm2d(filters_per_branch),
                activation,
                nn.Dropout(dropout_rate),
                DepthWiseConv2d(8, (1, num_channels), 2),
                nn.BatchNorm2d(filters_per_branch * 2),
                activation,
                nn.Dropout(dropout_rate),
            ) for scales_sample in scales_samples
        ])
        self.avg_pool1 = nn.AvgPool2d((4, 1))

        # ========================== BLOCK 2: INCEPTION ========================== #
        self.inception2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    len(scales_samples) * 2 * filters_per_branch,
                    filters_per_branch, (scales_sample // 4, 1),
                    bias=False,
                    padding="same"
                    # padding=((scales_sample // 4 - 1) // 2, 0)
                ) if torch.__version__ >= "1.9" else nn.Sequential(
                    CustomPad((0, 0, scales_sample // 8 - 1, scales_sample // 8, )),
                    nn.Conv2d(
                        len(scales_samples) * 2 * filters_per_branch,
                        filters_per_branch, (scales_sample // 4, 1),
                        bias=False
                    )
                ),
                nn.BatchNorm2d(filters_per_branch),
                activation,
                nn.Dropout(dropout_rate),
            ) for scales_sample in scales_samples
        ])

        self.avg_pool2 = nn.AvgPool2d((2, 1))

        # ============================ BLOCK 3: OUTPUT =========================== #
        self.output = nn.Sequential(
            nn.Conv2d(
                24, filters_per_branch * len(scales_samples) // 2, (8, 1),
                bias=False, padding='same'
            ) if torch.__version__ >= "1.9" else nn.Sequential(
                CustomPad((0, 0, 4, 3)),
                nn.Conv2d(
                    24, filters_per_branch * len(scales_samples) // 2, (8, 1),
                    bias=False
                )
            ),
            nn.BatchNorm2d(filters_per_branch * len(scales_samples) // 2),
            activation,
            nn.AvgPool2d((2, 1)),
            nn.Dropout(dropout_rate),

            nn.Conv2d(
                12, filters_per_branch * len(scales_samples) // 4, (4, 1),
                bias=False, padding='same'
            ) if torch.__version__ >= "1.9" else nn.Sequential(
                CustomPad((0, 0, 2, 1)),
                nn.Conv2d(
                    12, filters_per_branch * len(scales_samples) // 4, (4, 1),
                    bias=False
                )
            ),
            nn.BatchNorm2d(filters_per_branch * len(scales_samples) // 4),
            activation,
            nn.AvgPool2d((2, 1)),
            nn.Dropout(dropout_rate),
        )
        self.cls = nn.Sequential(
            nn.Linear(4 * 1 * 6, num_classes),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = torch.cat([net(x) for net in self.inception1], dim=1)
        x = self.avg_pool1(x)
        x = torch.cat([net(x) for net in self.inception2], dim=1)
        x = self.avg_pool2(x)
        x = self.output(x)
        x = torch.flatten(x, start_dim=1)
        return self.cls(x)