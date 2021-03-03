import numpy as np
import torch
import cv2


class PadandRandomCrop(object):
    '''
    Input tensor is expected to have shape of (H, W, 3)
    '''
    def __init__(self, border=4, cropsize=(32, 32)):
        self.border = border
        self.cropsize = cropsize

    def __call__(self, im):
        borders = [(self.border, self.border), (self.border, self.border), (0, 0)]  # input is (h, w, c)
        convas = np.pad(im, borders, mode='reflect')
        H, W, C = convas.shape
        h, w = self.cropsize
        dh, dw = max(0, H-h), max(0, W-w)
        sh, sw = np.random.randint(0, dh), np.random.randint(0, dw)
        out = convas[sh:sh+h, sw:sw+w, :]
        return out


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, im):
        if np.random.rand() < self.p:
            im = im[:, ::-1, :]
        return im


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, im):
        im = cv2.resize(im, self.size)
        return im


class Normalize(object):
    '''
    Inputs are pixel values in range of [0, 255], channel order is 'rgb'
    '''
    def __init__(self, mean, std):
        self.mean = np.array(mean, np.float32).reshape(1, 1, -1)
        self.std = np.array(std, np.float32).reshape(1, 1, -1)

    def __call__(self, im):
        if len(im.shape) == 4:
            mean, std = self.mean[None, ...], self.std[None, ...]
        elif len(im.shape) == 3:
            mean, std = self.mean, self.std
        im = im.astype(np.float32) / 255.
        #  im = (im.astype(np.float32) / 127.5) - 1
        im -= mean
        im /= std
        return im


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, im):
        if len(im.shape) == 4:
            return torch.from_numpy(im.transpose(0, 3, 1, 2))
        elif len(im.shape) == 3:
            return torch.from_numpy(im.transpose(2, 0, 1))


class Compose(object):
    def __init__(self, ops):
        self.ops = ops

    def __call__(self, im):
        for op in self.ops:
            im = op(im)
        return im
