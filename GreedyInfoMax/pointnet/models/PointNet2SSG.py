from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn

from PointNet2.pointnet2.pointnet2_modules import PointnetSAModule
from GreedyInfoMax.pointnet.models.PointNetEncoder import PointNetEncoder

class Pointnet2SSG(nn.Module):
    r"""
        PointNet2 with single-scale grouping

        Parameters
        ----------
        opt:
            options struct
        input_channels: int = 3
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        point_cloud_grouper: nn.Module
            Module used to divide point clouds into multiple subsets to use in the Greedy InfoMax learning algorithm
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, opt, input_channels=0, point_cloud_grouper=None, use_xyz=True):
        super(Pointnet2SSG, self).__init__()

        self.opt = opt

        npoints = [512 if not opt.loss == 'info_nce' else 128,
                   128 if not opt.loss == 'info_nce' else 32,
                   None]
        radii = [0.2,
                 0.4,
                 None]
        nsamples = [32,
                    64,
                    None]
        mlps = [[input_channels, 64, 64, 128],
                [128, 128, 128, 256],
                [256, 256, 512, 1024]]

        sa_modules = [PointnetSAModule(
                    npoint=npoints[i],
                    radius=radii[i],
                    nsample=nsamples[i],
                    mlp=mlps[i],
                    use_xyz=use_xyz,
                    bn=False,
            ) for i in range(len(npoints))]

        self.encoder = nn.ModuleList()

        if opt.model_splits == 1:
            self.encoder.append(
                PointNetEncoder(opt,
                    sa_modules, mlps[-1][-1], 0, point_cloud_grouper,)
            )
        elif opt.model_splits == 3:
            for i in range(3):
                self.encoder.append(
                    PointNetEncoder(opt,
                        [sa_modules[i]], mlps[i][-1], i, point_cloud_grouper if i == 0 else None)
                )
        else:
            raise NotImplementedError

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def set_calc_loss(self, calc_loss):
        for module in self.encoder:
            module.set_calc_loss(calc_loss)

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """

        if self.opt.device.type != "cpu":
            cur_device = pointcloud.get_device()
        else:
            cur_device = self.opt.device

        loss = torch.zeros(1, self.opt.model_splits, device=cur_device)  # first dimension for multi-GPU training

        xyz, features = self._break_up_pc(pointcloud)
        failed_groups = None

        for im, module in enumerate(self.encoder):
            new_xyz, new_features, cur_loss, failed_groups = module(xyz, features, failed_groups)

            if self.opt.loss == 'info_nce':  # Detach gradients if optimizing locally
                if new_xyz is None:
                    xyz = None
                else:
                    xyz = new_xyz.detach()
                features = new_features.detach()
            else:
                xyz = new_xyz
                features = new_features

            if cur_loss is not None:
                loss[:, im] = cur_loss

        return loss, features
