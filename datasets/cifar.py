import os.path as osp
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import transform as T
from datasets.randaugment import RandomAugment
from datasets.sampler import RandomSampler, BatchSampler

class TwoCropsTransform:
    """Take 2 random augmentations of one image."""

    def __init__(self,trans_weak,trans_strong):       
        self.trans_weak = trans_weak
        self.trans_strong = trans_strong
    def __call__(self, x):
        x1 = self.trans_weak(x)
        x2 = self.trans_strong(x)
        return [x1, x2]
    
class ThreeCropsTransform:
    """Take 3 random augmentations of one image."""

    def __init__(self,trans_weak,trans_strong0,trans_strong1):       
        self.trans_weak = trans_weak
        self.trans_strong0 = trans_strong0
        self.trans_strong1 = trans_strong1
    def __call__(self, x):
        x1 = self.trans_weak(x)
        x2 = self.trans_strong0(x)
        x3 = self.trans_strong1(x)
        return [x1, x2, x3]

def load_data_train(L=250, dataset='CIFAR10', dspth='./data'):
    if dataset == 'CIFAR10':
        datalist = [
            osp.join(dspth, 'cifar-10-batches-py', 'data_batch_{}'.format(i + 1))
            for i in range(5)
        ]
        n_class = 10
        assert L in [10, 20, 40, 80, 250, 4000]
    elif dataset == 'CIFAR100':
        datalist = [
            osp.join(dspth, 'cifar-100-python', 'train')]
        n_class = 100
        assert L in [25, 400, 2500, 10000]

    data, labels = [], []
    for data_batch in datalist:
        with open(data_batch, 'rb') as fr:
            entry = pickle.load(fr, encoding='latin1')
            lbs = entry['labels'] if 'labels' in entry.keys() else entry['fine_labels']
            data.append(entry['data'])
            labels.append(lbs)
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    n_labels = L // n_class
    data_x, label_x, data_u, label_u = [], [], [], []
    for i in range(n_class):
        indices = np.where(labels == i)[0]
        np.random.shuffle(indices)
        inds_x, inds_u = indices[:n_labels], indices[n_labels:]
        data_x += [
            data[i].reshape(3, 32, 32).transpose(1, 2, 0)
            for i in inds_x
        ]
        label_x += [labels[i] for i in inds_x]
        data_u += [
            data[i].reshape(3, 32, 32).transpose(1, 2, 0)
            for i in inds_u
        ]
        label_u += [labels[i] for i in inds_u]
    return data_x, label_x, data_u, label_u


def load_data_val(dataset, dspth='./data'):
    if dataset == 'CIFAR10':
        datalist = [
            osp.join(dspth, 'cifar-10-batches-py', 'test_batch')
        ]
    elif dataset == 'CIFAR100':
        datalist = [
            osp.join(dspth, 'cifar-100-python', 'test')
        ]

    data, labels = [], []
    for data_batch in datalist:
        with open(data_batch, 'rb') as fr:
            entry = pickle.load(fr, encoding='latin1')
            lbs = entry['labels'] if 'labels' in entry.keys() else entry['fine_labels']
            data.append(entry['data'])
            labels.append(lbs)
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    data = [
        el.reshape(3, 32, 32).transpose(1, 2, 0)
        for el in data
    ]
    return data, labels


def compute_mean_var():
    data_x, label_x, data_u, label_u = load_data_train()
    data = data_x + data_u
    data = np.concatenate([el[None, ...] for el in data], axis=0)

    mean, var = [], []
    for i in range(3):
        channel = (data[:, :, :, i].ravel() / 127.5) - 1
        #  channel = (data[:, :, :, i].ravel() / 255)
        mean.append(np.mean(channel))
        var.append(np.std(channel))

    print('mean: ', mean)
    print('var: ', var)



class Cifar(Dataset):
    def __init__(self, dataset, data, labels, mode):
        super(Cifar, self).__init__()
        self.data, self.labels = data, labels
        self.mode = mode
        assert len(self.data) == len(self.labels)
        if dataset == 'CIFAR10':
            mean, std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
        elif dataset == 'CIFAR100':
            mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

        trans_weak = T.Compose([
            T.Resize((32, 32)),
            T.PadandRandomCrop(border=4, cropsize=(32, 32)),
            T.RandomHorizontalFlip(p=0.5),
            T.Normalize(mean, std),
            T.ToTensor(),
        ])
        trans_strong0 = T.Compose([
            T.Resize((32, 32)),
            T.PadandRandomCrop(border=4, cropsize=(32, 32)),
            T.RandomHorizontalFlip(p=0.5),
            RandomAugment(2, 10),
            T.Normalize(mean, std),
            T.ToTensor(),
        ])        
        trans_strong1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),     
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),        
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])                    
        if self.mode == 'train_x':
            self.trans = trans_weak
        elif self.mode == 'train_u_comatch':
            self.trans = ThreeCropsTransform(trans_weak, trans_strong0, trans_strong1)               
        elif self.mode == 'train_u_fixmatch':
            self.trans = TwoCropsTransform(trans_weak, trans_strong0)    
        else:  
            self.trans = T.Compose([
                T.Resize((32, 32)),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])

    def __getitem__(self, idx):
        im, lb = self.data[idx], self.labels[idx]
        return self.trans(im), lb

    def __len__(self):
        leng = len(self.data)
        return leng


def get_train_loader(dataset, batch_size, mu, n_iters_per_epoch, L, root='data', method='comatch'):
    data_x, label_x, data_u, label_u = load_data_train(L=L, dataset=dataset, dspth=root)

    ds_x = Cifar(
        dataset=dataset,
        data=data_x,
        labels=label_x,
        mode='train_x'
    )  # return an iter of num_samples length (all indices of samples)
    sampler_x = RandomSampler(ds_x, replacement=True, num_samples=n_iters_per_epoch * batch_size)
    batch_sampler_x = BatchSampler(sampler_x, batch_size, drop_last=True)  # yield a batch of samples one time
    dl_x = torch.utils.data.DataLoader(
        ds_x,
        batch_sampler=batch_sampler_x,
        num_workers=2,
        pin_memory=True
    )
    ds_u = Cifar(
        dataset=dataset,
        data=data_u,
        labels=label_u,
        mode='train_u_%s'%method
    )
    sampler_u = RandomSampler(ds_u, replacement=True, num_samples=mu * n_iters_per_epoch * batch_size)
    #sampler_u = RandomSampler(ds_u, replacement=False)
    batch_sampler_u = BatchSampler(sampler_u, batch_size * mu, drop_last=True)
    dl_u = torch.utils.data.DataLoader(
        ds_u,
        batch_sampler=batch_sampler_u,
        num_workers=2,
        pin_memory=True
    )
    return dl_x, dl_u


def get_val_loader(dataset, batch_size, num_workers, pin_memory=True, root='data'):
    data, labels = load_data_val(dataset, dspth=root)
    ds = Cifar(
        dataset=dataset,
        data=data,
        labels=labels,
        mode='test'
    )
    dl = torch.utils.data.DataLoader(
        ds,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return dl


