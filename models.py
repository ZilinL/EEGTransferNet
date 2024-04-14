import torch
import torch.nn as nn
from transfer_losses import TransferLoss
import backbones
from center_alignment import CenterAlignment
from riemann_mean import *

##只在特征层进行迁移
class TransferNet(nn.Module):
    def __init__(self, num_class, base_net='eegnet', transfer_loss='mmd', max_iter=1000, **kwargs):
        super(TransferNet, self).__init__()
        self.num_class = num_class
        self.base_network = backbones.get_backbone(base_net)
        self.transfer_loss = transfer_loss
        feature_dim = self.base_network.output_num(**kwargs)
        
        self.classifier_layer = nn.Linear(feature_dim, num_class)
        transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "max_iter": max_iter,
            "num_class": num_class
        }
        self.adapt_loss = TransferLoss(**transfer_loss_args)
        self.criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([kwargs['loss_weight'][0],1])).float()) #weight=torch.from_numpy(np.array([kwargs['loss_weight'][0],1])).float()

    def forward(self, source, target, source_label):
        source = self.base_network(source)
        target = self.base_network(target)

        # classification
        source_clf = self.classifier_layer(source)
        clf_loss = self.criterion(source_clf, source_label)
        # transfer
        kwargs = {}
        if self.transfer_loss == "lmmd":
            kwargs['source_label'] = source_label
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == "daan":
            source_clf = self.classifier_layer(source)
            kwargs['source_logits'] = torch.nn.functional.softmax(source_clf, dim=1)
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == 'bnm':
            tar_clf = self.classifier_layer(target)
            target = nn.Softmax(dim=1)(tar_clf)
        
        transfer_loss = self.adapt_loss(source, target, **kwargs)
        return clf_loss, transfer_loss
    
    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
        ]
        # Loss-dependent
        if self.transfer_loss == "adv":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == "daan":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
            params.append(
                {'params': self.adapt_loss.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params

    def predict(self, x):
        features = self.base_network(x)
        self.feature = features
        clf = self.classifier_layer(features)
        return clf

    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == "daan":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass


##在head层、特征层进行迁移
class EEGTransferNet(nn.Module):
    def __init__(self, num_class, base_net='eegtransfernet', transfer_loss='lmmd', max_iter=1000, **kwargs):
        super(EEGTransferNet, self).__init__()
        self.num_class = num_class
        self.filter_head = backbones.get_filter_head()
        self.base_network = backbones.get_backbone(base_net)
        self.transfer_loss = transfer_loss
        self.head_loss_type = 'jeffrey'  #lem和jeffrey两种
        feature_dim = self.base_network.output_num(**kwargs)
        self.classifier_layer = nn.Linear(feature_dim, num_class)
        self.mean_point = JefferyMean()   # LEMMean()和JefferyMean()两种

        transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "max_iter": max_iter,
            "num_class": num_class
        }
        head_loss_args = {
            "loss_type": self.head_loss_type,
        }       

        self.adapt_head_loss = TransferLoss(**head_loss_args) 
        self.adapt_loss = TransferLoss(**transfer_loss_args)
        self.criterion = torch.nn.CrossEntropyLoss() # weight=torch.from_numpy(np.array([kwargs['loss_weight'][0],1])).float()

    def forward(self, source, target, source_label):
        src_filtered_data = self.filter_head(source)
        source_mean = self.mean_point(src_filtered_data)
        source = self.base_network(src_filtered_data)

        tgt_filtered_data = self.filter_head(target)
        target_mean = self.mean_point(tgt_filtered_data)
        target = self.base_network(tgt_filtered_data)
       

        kwargs = {}

        # filtered data
        head_loss = self.adapt_head_loss(source_mean, target_mean, **kwargs)
        
        if self.transfer_loss == "lmmd":
            kwargs['source_label'] = source_label
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == "daan":
            source_clf = self.classifier_layer(source)
            kwargs['source_logits'] = torch.nn.functional.softmax(source_clf, dim=1)
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == 'bnm':
            tar_clf = self.classifier_layer(target)
            target = nn.Softmax(dim=1)(tar_clf)

        # classification
        source_clf = self.classifier_layer(source)
        clf_loss = self.criterion(source_clf, source_label)
        # transfer        
        transfer_loss = self.adapt_loss(source, target, **kwargs)
        return clf_loss, transfer_loss, head_loss
    
    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.filter_head.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
        ]
        # Loss-dependent
        if self.transfer_loss == "adv":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == "daan":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
            params.append(
                {'params': self.adapt_loss.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params

    def predict(self, x):
        filtered_data = self.filter_head(x)
        features = self.base_network(filtered_data)
        clf = self.classifier_layer(features)
        return clf

    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == "daan":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass


##只在head层进行迁移
class EEGTransferNet_head(nn.Module):
    def __init__(self, num_class, base_net='eegtransfernet', transfer_loss='mmd', max_iter=1000, **kwargs):
        super(EEGTransferNet_head, self).__init__()
        self.num_class = num_class
        self.filter_head = backbones.get_filter_head()
        self.base_network = backbones.get_backbone(base_net)
        self.transfer_loss = transfer_loss
        self.head_loss_type = 'jeffrey'
        feature_dim = self.base_network.output_num(**kwargs)
        self.classifier_layer = nn.Linear(feature_dim, num_class)
        self.mean_point = JefferyMean()   

        head_loss_args = {
            "loss_type": self.head_loss_type,
        }       

        self.adapt_head_loss = TransferLoss(**head_loss_args) 
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, source, target, source_label):
        src_filtered_data = self.filter_head(source)
        source_mean = self.mean_point(src_filtered_data)
        source = self.base_network(src_filtered_data)

        tgt_filtered_data = self.filter_head(target)
        target_mean = self.mean_point(tgt_filtered_data)
        target = self.base_network(tgt_filtered_data)
       

        kwargs = {}

        # filtered data
        head_loss = self.adapt_head_loss(source_mean, target_mean, **kwargs)

        # classification
        source_clf = self.classifier_layer(source)
        clf_loss = self.criterion(source_clf, source_label)
      
        return clf_loss, head_loss
    
    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.filter_head.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
        ]
        # Loss-dependent
        if self.transfer_loss == "adv":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == "daan":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
            params.append(
                {'params': self.adapt_loss.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params

    def predict(self, x):
        filtered_data = self.filter_head(x)
        features = self.base_network(filtered_data)
        clf = self.classifier_layer(features)
        return clf

    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == "daan":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass



#没有迁移
class EEGNet(nn.Module):
    def __init__(self, num_class, base_net='eegnet', **kwargs):
        super(EEGNet, self).__init__()
        self.num_class = num_class
        self.base_network = backbones.get_backbone(base_net)
        feature_dim = self.base_network.output_num(**kwargs)
        self.classifier_layer = nn.Linear(feature_dim, num_class)


    def forward(self, data):
        x = self.base_network(data)
        x = self.classifier_layer(x)
        return x
    
    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
        ]
        return params

    def predict(self, x):
        features = self.base_network(x)
        self.feature = features
        clf = self.classifier_layer(features)
        return clf
        


##在head层CA,特征层进行迁移
class EEGTransferNet_CA(nn.Module):
    def __init__(self, num_class, base_net='eegtransfernet', transfer_loss='lmmd', CA_type = 'euclid', max_iter=1000, filterbank='filter2',**kwargs):
        super(EEGTransferNet_CA, self).__init__()
        self.num_class = num_class
        self.filter_head = backbones.get_filter_head(filterbank)
        self.CA = CenterAlignment(CA_type)
        self.base_network = backbones.get_backbone(base_net)
        self.transfer_loss = transfer_loss
        feature_dim = self.base_network.output_num(**kwargs)
        self.classifier_layer = nn.Linear(feature_dim, num_class)
        self.mean_point = JefferyMean()   # LEMMean()和JefferyMean()两种

        transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "max_iter": max_iter,
            "num_class": num_class
        }
      
        self.adapt_loss = TransferLoss(**transfer_loss_args)
        self.criterion = torch.nn.CrossEntropyLoss() # weight=torch.from_numpy(np.array([kwargs['loss_weight'][0],1])).float()

    def forward(self, source, target, source_label):
        src_filtered_data = self.filter_head(source)
        source_CA = self.CA(src_filtered_data)
        source = self.base_network(source_CA)

        tgt_filtered_data = self.filter_head(target)
        target_CA = self.CA(tgt_filtered_data)
        target = self.base_network(target_CA)
       

        kwargs = {}

        # filtered data
        
        if self.transfer_loss == "lmmd":
            kwargs['source_label'] = source_label
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == "daan":
            source_clf = self.classifier_layer(source)
            kwargs['source_logits'] = torch.nn.functional.softmax(source_clf, dim=1)
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == 'bnm':
            tar_clf = self.classifier_layer(target)
            target = nn.Softmax(dim=1)(tar_clf)

        # classification
        source_clf = self.classifier_layer(source)
        clf_loss = self.criterion(source_clf, source_label)
        # transfer        
        transfer_loss = self.adapt_loss(source, target, **kwargs)
        return clf_loss, transfer_loss
    
    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.filter_head.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
        ]
        # Loss-dependent
        if self.transfer_loss == "adv":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == "daan":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
            params.append(
                {'params': self.adapt_loss.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params

    def predict(self, x):
        filtered_data = self.filter_head(x)
        data_CA = self.CA(filtered_data)
        features = self.base_network(data_CA)
        self.feature = features
        clf = self.classifier_layer(features)
        return clf

    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == "daan":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass



##只在head层CA
class EEGTransferNet_CAhead(nn.Module):
    def __init__(self, num_class, base_net='eegtransfernet', CA_type = 'euclid', filterbank='filter2', **kwargs): #riemann euclid logdet logeuclid
        super(EEGTransferNet_CAhead, self).__init__()
        self.num_class = num_class
        self.filter_head = backbones.get_filter_head(filterbank)
        self.CA = CenterAlignment(CA_type)
        self.base_network = backbones.get_backbone(base_net)
        feature_dim = self.base_network.output_num(**kwargs)
        self.classifier_layer = nn.Linear(feature_dim, num_class)

        self.criterion = torch.nn.CrossEntropyLoss() # weight=torch.from_numpy(np.array([kwargs['loss_weight'][0],1])).float()

    def forward(self, source, target, source_label):
        src_filtered_data = self.filter_head(source)
        source_CA = self.CA(src_filtered_data)
        source = self.base_network(source_CA)

        tgt_filtered_data = self.filter_head(target)
        target_CA = self.CA(tgt_filtered_data)
        target = self.base_network(target_CA)

        # classification
        source_clf = self.classifier_layer(source)
        clf_loss = self.criterion(source_clf, source_label)
        # transfer        
        return clf_loss
    
    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.filter_head.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
        ]
        return params

    def predict(self, x):
        filtered_data = self.filter_head(x)
        data_CA = self.CA(filtered_data)
        features = self.base_network(data_CA)
        self.feature = features
        clf = self.classifier_layer(features)
        return clf

    def epoch_based_processing(self, *args, **kwargs):
            pass



##在head层固定，CA,特征层进行迁移
class EEGTransferNet_frozenHead_CA(nn.Module):
    def __init__(self, num_class, base_net='eegtransfernet', transfer_loss='lmmd', CA_type = 'euclid', max_iter=1000, **kwargs):
        super(EEGTransferNet_frozenHead_CA, self).__init__()
        self.num_class = num_class
        self.s_filter_head = backbones.get_filter_head()
        self.t_filter_head = backbones.get_filter_head()
        self.CA = CenterAlignment(CA_type)
        self.base_network = backbones.get_backbone(base_net)
        self.transfer_loss = transfer_loss
        feature_dim = self.base_network.output_num(**kwargs)
        self.classifier_layer = nn.Linear(feature_dim, num_class)
        self.mean_point = JefferyMean()   # LEMMean()和JefferyMean()两种

        transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "max_iter": max_iter,
            "num_class": num_class
        }
      
        self.adapt_loss = TransferLoss(**transfer_loss_args)
        self.criterion = torch.nn.CrossEntropyLoss() # weight=torch.from_numpy(np.array([kwargs['loss_weight'][0],1])).float()

    def forward(self, source, target, source_label):
        src_filtered_data = self.s_filter_head(source)
        source_CA = self.CA(src_filtered_data)
        source = self.base_network(source_CA)

        tgt_filtered_data = self.t_filter_head(target)
        target_CA = self.CA(tgt_filtered_data)
        target = self.base_network(target_CA)
       

        kwargs = {}

        # filtered data
        
        if self.transfer_loss == "lmmd":
            kwargs['source_label'] = source_label
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == "daan":
            source_clf = self.classifier_layer(source)
            kwargs['source_logits'] = torch.nn.functional.softmax(source_clf, dim=1)
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == 'bnm':
            tar_clf = self.classifier_layer(target)
            target = nn.Softmax(dim=1)(tar_clf)

        # classification
        source_clf = self.classifier_layer(source)
        clf_loss = self.criterion(source_clf, source_label)
        # transfer        
        transfer_loss = self.adapt_loss(source, target, **kwargs)
        return clf_loss, transfer_loss
    
    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.s_filter_head.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.t_filter_head.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
        ]
        # Loss-dependent
        if self.transfer_loss == "adv":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == "daan":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
            params.append(
                {'params': self.adapt_loss.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params

    def predict(self, x):
        filtered_data = self.t_filter_head(x)
        data_CA = self.CA(filtered_data)
        features = self.base_network(data_CA)
        clf = self.classifier_layer(features)
        return clf

    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == "daan":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass


##在backbone固定，CA,特征层进行迁移
class EEGTransferNet_frozenBackbone_CA(nn.Module):
    def __init__(self, num_class, base_net='eegtransfernet', transfer_loss='lmmd', CA_type = 'euclid', max_iter=1000, **kwargs):
        super(EEGTransferNet_frozenBackbone_CA, self).__init__()
        self.num_class = num_class
        self.s_filter_head = backbones.get_filter_head()
        self.t_filter_head = backbones.get_filter_head()
        self.CA = CenterAlignment(CA_type)
        self.s_base_network = backbones.get_backbone(base_net)
        self.t_base_network = backbones.get_backbone(base_net)
        self.transfer_loss = transfer_loss
        feature_dim = self.s_base_network.output_num(**kwargs)
        self.classifier_layer = nn.Linear(feature_dim, num_class)
        self.mean_point = JefferyMean()   # LEMMean()和JefferyMean()两种

        transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "max_iter": max_iter,
            "num_class": num_class
        }
      
        self.adapt_loss = TransferLoss(**transfer_loss_args)
        self.criterion = torch.nn.CrossEntropyLoss() # weight=torch.from_numpy(np.array([kwargs['loss_weight'][0],1])).float()

    def forward(self, source, target, source_label):
        src_filtered_data = self.s_filter_head(source)
        source_CA = self.CA(src_filtered_data)
        source = self.s_base_network(source_CA)

        tgt_filtered_data = self.t_filter_head(target)
        target_CA = self.CA(tgt_filtered_data)
        target = self.t_base_network(target_CA)
       

        kwargs = {}

        # filtered data
        
        if self.transfer_loss == "lmmd":
            kwargs['source_label'] = source_label
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == "daan":
            source_clf = self.classifier_layer(source)
            kwargs['source_logits'] = torch.nn.functional.softmax(source_clf, dim=1)
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == 'bnm':
            tar_clf = self.classifier_layer(target)
            target = nn.Softmax(dim=1)(tar_clf)

        # classification
        source_clf = self.classifier_layer(source)
        clf_loss = self.criterion(source_clf, source_label)
        # transfer        
        transfer_loss = self.adapt_loss(source, target, **kwargs)
        return clf_loss, transfer_loss
    
    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.s_filter_head.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.t_filter_head.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.s_base_network.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.t_base_network.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
        ]
        # Loss-dependent
        if self.transfer_loss == "adv":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == "daan":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
            params.append(
                {'params': self.adapt_loss.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params

    def predict(self, x):
        filtered_data = self.t_filter_head(x)
        data_CA = self.CA(filtered_data)
        t_features = self.t_base_network(data_CA)
        self.t_feature = t_features
        clf = self.classifier_layer(t_features)

        s_filtered_data = self.s_filter_head(x)
        s_data_CA = self.CA(s_filtered_data)
        s_features = self.s_base_network(s_data_CA)
        self.s_feature = s_features

        return clf

    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == "daan":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass

##########################################################################################################################
###################################################在filter的不同层加入fal模块##############################################
##########################################################################################################################
#卷积层第一层加入fal
class EEGTransferNet_CA_filter_1(nn.Module):
    def __init__(self, num_class, base_net='eegtransfernet1', transfer_loss='lmmd', CA_type = 'euclid', max_iter=1000, filterbank='filter1', **kwargs):
        super(EEGTransferNet_CA_filter_1, self).__init__()
        self.num_class = num_class
        self.filter_head = backbones.get_filter_head(filterbank)
        self.CA = CenterAlignment(CA_type)
        self.base_network = backbones.get_backbone(base_net)
        self.transfer_loss = transfer_loss
        feature_dim = self.base_network.output_num(**kwargs)
        self.classifier_layer = nn.Linear(feature_dim, num_class)
        self.mean_point = JefferyMean()   # LEMMean()和JefferyMean()两种

        transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "max_iter": max_iter,
            "num_class": num_class
        }
      
        self.adapt_loss = TransferLoss(**transfer_loss_args)
        self.criterion = torch.nn.CrossEntropyLoss() # weight=torch.from_numpy(np.array([kwargs['loss_weight'][0],1])).float()

    def forward(self, source, target, source_label):
        src_filtered_data = self.filter_head(source)
        source_CA = self.CA(src_filtered_data)
        source = self.base_network(source_CA)

        tgt_filtered_data = self.filter_head(target)
        target_CA = self.CA(tgt_filtered_data)
        target = self.base_network(target_CA)
       

        kwargs = {}

        # filtered data
        
        if self.transfer_loss == "lmmd":
            kwargs['source_label'] = source_label
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == "daan":
            source_clf = self.classifier_layer(source)
            kwargs['source_logits'] = torch.nn.functional.softmax(source_clf, dim=1)
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == 'bnm':
            tar_clf = self.classifier_layer(target)
            target = nn.Softmax(dim=1)(tar_clf)

        # classification
        source_clf = self.classifier_layer(source)
        clf_loss = self.criterion(source_clf, source_label)
        # transfer        
        transfer_loss = self.adapt_loss(source, target, **kwargs)
        return clf_loss, transfer_loss
    
    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.filter_head.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
        ]
        # Loss-dependent
        if self.transfer_loss == "adv":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == "daan":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
            params.append(
                {'params': self.adapt_loss.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params

    def predict(self, x):
        filtered_data = self.filter_head(x)
        data_CA = self.CA(filtered_data)
        features = self.base_network(data_CA)
        self.feature = features
        clf = self.classifier_layer(features)
        return clf

    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == "daan":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass

###################################最前面加入fal###########################################
##在head层CA,特征层进行迁移
class EEGTransferNet_CA_filter0(nn.Module):
    def __init__(self, num_class, base_net='eegtransfernet', transfer_loss='lmmd', CA_type = 'euclid', max_iter=1000, filterbank='filter2',**kwargs):
        super(EEGTransferNet_CA_filter0, self).__init__()
        self.num_class = num_class
        self.filter_head = backbones.get_filter_head(filterbank)
        self.CA = CenterAlignment(CA_type)
        self.base_network = backbones.get_backbone(base_net)
        self.transfer_loss = transfer_loss
        feature_dim = self.base_network.output_num(**kwargs)
        self.classifier_layer = nn.Linear(feature_dim, num_class)
        self.mean_point = JefferyMean()   # LEMMean()和JefferyMean()两种

        transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "max_iter": max_iter,
            "num_class": num_class
        }
      
        self.adapt_loss = TransferLoss(**transfer_loss_args)
        self.criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([kwargs['loss_weight'][0],1])).float()) # weight=torch.from_numpy(np.array([kwargs['loss_weight'][0],1])).float()

    def forward(self, source, target, source_label):
        source_CA = self.CA(source)
        source_CA = source_CA.permute(0, 2, 1, 3)
        src_filtered_data = self.filter_head(source_CA)
        source = self.base_network(src_filtered_data)

        target_CA = self.CA(target)
        target_CA = target_CA.permute(0, 2, 1, 3)
        tgt_filtered_data = self.filter_head(target_CA)
        target = self.base_network(tgt_filtered_data)
       

        kwargs = {}

        # filtered data
        
        if self.transfer_loss == "lmmd":
            kwargs['source_label'] = source_label
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == "daan":
            source_clf = self.classifier_layer(source)
            kwargs['source_logits'] = torch.nn.functional.softmax(source_clf, dim=1)
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == 'bnm':
            tar_clf = self.classifier_layer(target)
            target = nn.Softmax(dim=1)(tar_clf)

        # classification
        source_clf = self.classifier_layer(source)
        clf_loss = self.criterion(source_clf, source_label)
        # transfer        
        transfer_loss = self.adapt_loss(source, target, **kwargs)
        return clf_loss, transfer_loss
    
    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.filter_head.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
        ]
        # Loss-dependent
        if self.transfer_loss == "adv":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == "daan":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
            params.append(
                {'params': self.adapt_loss.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params

    def predict(self, x):
        filtered_data = self.filter_head(x)
        data_CA = self.CA(filtered_data)
        features = self.base_network(data_CA)
        self.feature = features
        clf = self.classifier_layer(features)
        return clf

    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == "daan":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass

###################################第三个卷积网络后加入fal###########################################
##在head层CA,特征层进行迁移
class EEGTransferNet_CA_filter3(nn.Module):
    def __init__(self, num_class, base_net='eegtransfernet3', transfer_loss='lmmd', CA_type = 'euclid', max_iter=1000, filterbank='filter3',**kwargs):
        super(EEGTransferNet_CA_filter3, self).__init__()
        self.num_class = num_class
        self.filter_head = backbones.get_filter_head(filterbank)
        self.CA = CenterAlignment(CA_type)
        self.base_network = backbones.get_backbone(base_net)
        self.transfer_loss = transfer_loss
        feature_dim = self.base_network.output_num(**kwargs)
        self.classifier_layer = nn.Linear(feature_dim, num_class)
        self.mean_point = JefferyMean()   # LEMMean()和JefferyMean()两种

        transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "max_iter": max_iter,
            "num_class": num_class
        }
      
        self.adapt_loss = TransferLoss(**transfer_loss_args)
        self.criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([kwargs['loss_weight'][0],1])).float()) # weight=torch.from_numpy(np.array([kwargs['loss_weight'][0],1])).float()

    def forward(self, source, target, source_label):
        src_filtered_data = self.filter_head(source)
        src_filtered_data = src_filtered_data.permute(0, 2, 1, 3)
        source_CA = self.CA(src_filtered_data)
        source = self.base_network(source_CA)

        tgt_filtered_data = self.filter_head(target)
        tgt_filtered_data = tgt_filtered_data.permute(0, 2, 1, 3)
        target_CA = self.CA(tgt_filtered_data)
        target = self.base_network(target_CA)      

        kwargs = {}

        # filtered data
        
        if self.transfer_loss == "lmmd":
            kwargs['source_label'] = source_label
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == "daan":
            source_clf = self.classifier_layer(source)
            kwargs['source_logits'] = torch.nn.functional.softmax(source_clf, dim=1)
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == 'bnm':
            tar_clf = self.classifier_layer(target)
            target = nn.Softmax(dim=1)(tar_clf)

        # classification
        source_clf = self.classifier_layer(source)
        clf_loss = self.criterion(source_clf, source_label)
        # transfer        
        transfer_loss = self.adapt_loss(source, target, **kwargs)
        return clf_loss, transfer_loss
    
    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.filter_head.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
        ]
        # Loss-dependent
        if self.transfer_loss == "adv":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == "daan":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
            params.append(
                {'params': self.adapt_loss.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params

    def predict(self, x):
        filtered_data = self.filter_head(x)
        data_CA = self.CA(filtered_data)
        features = self.base_network(data_CA)
        self.feature = features
        clf = self.classifier_layer(features)
        return clf

    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == "daan":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass