import torch
import torchvision.transforms
import torchvision
import os
import numpy as np
from copy import copy

from GreedyInfoMax.pointnet.data.ModelNet40Dataset import ModelNet40Cls
import GreedyInfoMax.pointnet.data.data_utils as data_utils


def get_dataloader(opt):
    if opt.dataset == "modelnet40":
        unsupervised_loader, \
        unsupervised_dataset, \
        train_loader, \
        train_dataset, \
        test_loader, \
        test_dataset = get_modelnet40_dataloaders(
            opt
        )
    else:
        raise Exception("Invalid option")

    return (
        unsupervised_loader,
        unsupervised_dataset,
        train_loader,
        train_dataset,
        test_loader,
        test_dataset,
    )

def get_modelnet40_dataloaders(opt):
    base_folder = os.path.join(opt.data_input_dir, "modelnet40")

    transforms = torchvision.transforms.transforms.Compose(
        [
            data_utils.PointcloudToTensor(),
            data_utils.PointcloudRotate(axis=np.array([1, 0, 0])),
            data_utils.PointcloudScale(),
            data_utils.PointcloudTranslate(),
            data_utils.PointcloudJitter(),
        ]
    )

    modelnet40_train_dataset = ModelNet40Cls(base_folder, opt.num_points, train=True, transforms=transforms)
    modelnet40_test_dataset = ModelNet40Cls(base_folder, opt.num_points, train=False, transforms=None)

    modelnet40_unsupervised_dataset = copy(modelnet40_train_dataset)

    # Split between supervised and unsupervised training data
    modelnet40_unsupervised_dataset.points = modelnet40_unsupervised_dataset.points[
                                             :opt.num_unsupervised_training_samples]
    modelnet40_unsupervised_dataset.labels = modelnet40_unsupervised_dataset.labels[
                                             :opt.num_unsupervised_training_samples]
    modelnet40_train_dataset.points = modelnet40_train_dataset.points[
                                             opt.num_unsupervised_training_samples:]
    modelnet40_train_dataset.labels = modelnet40_train_dataset.labels[
                                             opt.num_unsupervised_training_samples:]

    # default dataset loaders
    train_loader = torch.utils.data.DataLoader(
        modelnet40_train_dataset,
        batch_size=opt.batch_size_multiGPU,
        shuffle=True,
        num_workers=opt.num_workers
    )

    unsupervised_loader = torch.utils.data.DataLoader(
        modelnet40_unsupervised_dataset,
        batch_size=opt.batch_size_multiGPU,
        shuffle=True,
        num_workers=opt.num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        modelnet40_test_dataset,
        batch_size=opt.batch_size_multiGPU,
        shuffle=False,
        num_workers=opt.num_workers
    )

    return unsupervised_loader, modelnet40_unsupervised_dataset, \
           train_loader, modelnet40_train_dataset, \
           test_loader, modelnet40_test_dataset