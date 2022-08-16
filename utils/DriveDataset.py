import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pylab
import os
import time
from tqdm import tqdm, trange

from skimage import io
import matplotlib.pyplot as plt
import torchvision as tv

class DriveDataset(tc.utils.data.Dataset):
    """Blood vessel segmentation dataset."""

    def __init__(self, task, root_dir, transform=None):
        """
        Args:
            task (string): current task, either "training" or "test".
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self._build_list_images(task)
    
    def _build_list_images(self, task):
        self.im_paths = []
        self.gt_paths = []
        
        for i in range(21, 41):
            self.im_paths.append(os.path.join(task, "images", "{0}_training.tif".format(i)))
            self.gt_paths.append(os.path.join(task, "1st_manual", "{0}_manual1.gif".format(i)))

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        if tc.is_tensor(idx):
            idx = idx.tolist()

        im_name = os.path.join(self.root_dir, self.im_paths[idx])
        gt_name = os.path.join(self.root_dir, self.gt_paths[idx])
        image = io.imread(im_name)
        label = io.imread(gt_name)
        print(np.unique(image), np.unique(label))
        
        seed = tc.random.seed()
        
        if self.transform:
            tc.random.manual_seed(seed)
            image = self.transform(image)
            tc.random.manual_seed(seed)
            label = self.transform(label)
        return image, label

transform_valid = tv.transforms.Compose([tv.transforms.ToTensor(),
                              # tv.transforms.Normalize(0.5, 0.5),
                              # tv.transforms.Grayscale(num_output_channels=1),
                              tv.transforms.RandomCrop((500, 500))
                              # tv.transforms.CenterCrop((500, 500))
                              ])