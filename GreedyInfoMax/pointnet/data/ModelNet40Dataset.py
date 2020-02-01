"""
    Code originally written by Erik Wijmans (https://github.com/erikwijmans/Pointnet2_PyTorch)
    Adapted by Jesper Westell
"""

from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.utils.data as data
import numpy as np
import os
import h5py
import subprocess
import shlex

def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip()[5:] for line in f]


def _load_data_file(name):
    f = h5py.File(name)
    data = f["data"][:]
    label = f["label"][:]
    return data, label


class ModelNet40Cls(data.Dataset):
    def __init__(self, data_dir, num_points=1024, transforms=None, train=True, download=True):
        super().__init__()

        self.transforms = transforms

        self.url = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"

        if download and not os.path.exists(data_dir):
            zipfile = os.path.join(os.path.dirname(data_dir), os.path.basename(self.url))
            subprocess.check_call(
                shlex.split("curl {} -o {}".format(self.url, zipfile))
            )

            subprocess.check_call(
                shlex.split("unzip {} -d {}".format(zipfile, data_dir))
            )

            subprocess.check_call(shlex.split("rm {}".format(zipfile)))

        unzip_dir = os.path.join(data_dir, os.path.splitext(os.path.basename(self.url))[0])

        self.train = train
        if self.train:
            self.files = _get_data_files(os.path.join(unzip_dir, "train_files.txt"))
        else:
            self.files = _get_data_files(os.path.join(unzip_dir, "test_files.txt"))

        point_list, label_list = [], []
        for f in self.files:
            points, labels = _load_data_file(os.path.join(data_dir, f))
            point_list.append(points)
            label_list.append(labels)

        self.points = np.concatenate(point_list, 0)
        self.labels = np.concatenate(label_list, 0)
        self.set_num_points(num_points)

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.num_points)
        np.random.shuffle(pt_idxs)

        current_points = self.points[idx, pt_idxs].copy()
        label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)

        if self.transforms is not None:
            current_points = self.transforms(current_points)

        return current_points, label

    def __len__(self):
        return self.points.shape[0]

    def set_num_points(self, pts):
        self.num_points = min(self.points.shape[1], pts)

    def randomize(self):
        pass
