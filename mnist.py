import os

import paddle
import numpy as np
from paddle.vision.datasets import MNIST, Cifar10
from PIL import Image

import utils

class Mnist(paddle.io.Dataset):
    """
    Toy Demo
    """
    def __init__(self, split, transforms=None):
        """
        Init
        """
        super().__init__()
        self.transforms = transforms
        self.split = split

        assert split in ['train', 'test']

        samples = MNIST(backend='cv2', mode=split)

        x_load = np.stack([sample[0] for sample in samples])
        y_load = np.stack([sample[1] for sample in samples])[:, 0]

        self.x = x_load/255.
        self.y = y_load
        # print(self.x.shape)
        # 28->32
        pad_amount = ((0, 0), (2, 2), (2, 2))
        self.x = np.pad(self.x, pad_amount, 'constant')

        # let's get some shapes to understand what we loaded.
        print('shape of X: {}, y: {}'.format(self.x.shape, self.y.shape))

    def __getitem__(self, idx):
        """
        get item
        """
        img = self.x[idx, ..., None].transpose((1,2,0))
        label = self.y[idx]

        if self.transforms:
            img = self.transforms(img)

        coord = utils.get_coordinate_grid(img.shape[1]).astype("float32")
        return img, coord, label, idx

    def __len__(self):
        return len(self.x)



class Cifar(paddle.io.Dataset):
    """
    Cifar10
    """
    def __init__(self, split, transforms=None):
        """
        Init
        """
        super().__init__()
        self.transforms = transforms
        self.split = split

        assert split in ['train', 'test']

        samples = Cifar10(backend='pil', mode=split, transform=self.transforms)
        x_load = np.stack([sample[0] for sample in samples])
        # print(np.stack([sample[1] for sample in samples]).shape)
        # (50000, )
        y_load = np.stack([sample[1] for sample in samples])
        #print('x_load')
        self.x = x_load
        self.y = y_load
        # Already 32*32, No need to pad
        # pad_amount = ((0, 0), (2, 2), (2, 2))
        # print(self.x.shape)
        # self.x = np.pad(self.x, pad_amount, 'constant')

        # let's get some shapes to understand what we loaded.
        print('shape of X: {}, y: {}'.format(self.x.shape, self.y.shape))

    def __getitem__(self, idx):
        """
        get item
        """
        # print('getting')
        img = self.x[idx].transpose(1,2,0)  # N, H, W, C
        label = self.y[idx]
        # print(img.shape)
        # if self.transforms:
        #     img = self.transforms(img)

        coord = utils.get_coordinate_grid(img.shape[1]).astype("float32")
        return img, coord, label, idx

    def __len__(self):
        return len(self.x)