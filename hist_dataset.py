from __future__ import print_function
import tarfile
import numpy as np
import six
from PIL import Image
from six.moves import cPickle as pickle
from scipy.io import loadmat
import paddle
from paddle.io import Dataset
from paddle.dataset.common import _check_exists_and_download

__all__ = []


class corel_10k_train(Dataset):

    def __init__(self,
                 transform=None, ):
        self.transform = transform
        # read dataset into memory
        self._load_data()
        self.dtype = paddle.get_default_dtype()

    def _load_data(self):
        self.data = []
        output = open('train_data.pkl', 'rb')
        self.data = pickle.load(output)

    def __getitem__(self, idx):
        image, label = self.data[idx]

        return image.astype(self.dtype), np.array(label).astype('int64')

    def __len__(self):
        return len(self.data)


class corel_10k_test(Dataset):

    def __init__(self,
                 transform=None, ):
        self.transform = transform
        # read dataset into memory
        self._load_data()
        self.dtype = paddle.get_default_dtype()

    def _load_data(self):
        self.data = []
        output = open('train_data.pkl', 'rb')
        self.data = pickle.load(output)

    def __getitem__(self, idx):
        image, label = self.data[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image.astype(self.dtype), np.array(label).astype('int64')

    def __len__(self):
        return len(self.data)