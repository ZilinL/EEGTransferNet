import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from scipy.io import loadmat


def load_data(data_folder, BATCH_SIZE, SHUFFLE, NUM_WORKERS, **kwargs):
    data_mat = loadmat(data_folder)
    data = data_mat['x'] # (channels, trial_length, samples)
    num_channels = data.shape[0]
    trial_length = data.shape[1]
    label = data_mat['y']
    n_class = len(np.unique(label))
    dataset = EEGDataset(data, label)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS)
    loss_weight = sum(label)/data.shape[2]
    return data_loader, n_class, num_channels, trial_length, loss_weight

def load_splitted_data(data_folder, ratio):
    data_mat = loadmat(data_folder)
    data = data_mat['X'] # (channels, trial_length, samples)
    num_channels = data.shape[0]
    trial_length = data.shape[1]
    n_trials = data.shape[2]
    label = data_mat['y']
    n_class = len(np.unique(label))

    split_point = int(n_trials * ratio)

    part0_x, part0_y, part1_x, part1_y = data[:,:,:split_point], label[:split_point], data[:,:,split_point:], label[split_point:]
    part0_dataset = EEGDataset(part0_x, part0_y)
    part1_dataset = EEGDataset(part1_x, part1_y)
    return part0_dataset, part1_dataset, n_class, num_channels, trial_length



class EEGDataset(Dataset):
    def __init__(self, x, y):
        self.data = torch.from_numpy(x).float()
        self.data = self.data.unsqueeze(0)
        self.label = torch.LongTensor(y)
        self.label = self.label.squeeze(1)

    def __getitem__(self, index):
        return self.data[:,:,:,index], self.label[index]
    
    def __len__(self):
        return self.data.shape[3]