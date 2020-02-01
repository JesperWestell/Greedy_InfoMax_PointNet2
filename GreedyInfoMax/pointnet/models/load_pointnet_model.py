import torch

from GreedyInfoMax.utils import model_utils
from GreedyInfoMax.pointnet.models.model_utils import PointCloudGrouper
from GreedyInfoMax.pointnet.models.PointNet2SSG import Pointnet2SSG
from GreedyInfoMax.pointnet.models.ClassificationModel import ClassificationModel


def load_model_and_optimizer(opt, num_GPU=None, reload_model=False):
    point_cloud_grouper = PointCloudGrouper(opt)

    model = Pointnet2SSG(opt, point_cloud_grouper=point_cloud_grouper)

    optimizer = []
    print(opt.loss)
    if opt.model_splits == 1 or opt.loss == "supervised":
        optimizer.append(
            torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
        )
    elif opt.model_splits >= 3:
        # use separate optimizer for each module, so gradients don't get mixed up
        for idx, layer in enumerate(model.encoder):
            optimizer.append(torch.optim.Adam(layer.parameters(), lr=opt.learning_rate))
    else:
        raise NotImplementedError

    model, num_GPU = model_utils.distribute_over_GPUs(opt, model, num_GPU=num_GPU)

    model, optimizer = model_utils.reload_weights(
        opt, model, optimizer, reload_model=reload_model
    )

    return model, optimizer

def load_classification_model(opt):
    model = ClassificationModel().to(opt.device)
    return model