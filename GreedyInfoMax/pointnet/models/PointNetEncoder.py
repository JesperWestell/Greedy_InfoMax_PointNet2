import torch
import torch.nn as nn
from GreedyInfoMax.pointnet.models.InfoNCE_Loss import InfoNCE_Loss

class PointNetEncoder(nn.Module):
    def __init__(self, opt, SA_modules: list, feature_length: int, encoder_num: int, point_cloud_grouper=None):
        super(PointNetEncoder, self).__init__()
        self.opt = opt
        self.SA_modules = nn.ModuleList(SA_modules)
        self.encoder_num = encoder_num
        self.calc_loss = opt.loss == 'info_nce'
        self.grouper = point_cloud_grouper
        if self.calc_loss and self.encoder_num == 0:
            assert self.grouper is not None, "Needs to have centers for each sub point cloud defined!"

        # Always add loss for parameter loading reasons, but might not use it
        self.loss = InfoNCE_Loss(opt,
                                 in_channels=feature_length,
                                 out_channels=feature_length)
        self.failed_groups = None

    def set_calc_loss(self, calc_loss):
        self.calc_loss = calc_loss

    def _patchify(self, x):
        x, failed_groups = self.grouper(xyz=x)
        return x, failed_groups

    def forward(self, xyz, features, failed_groups=None):
        if self.calc_loss and self.encoder_num == 0:
            assert features is None  # Should only do this at first layer
            xyz, failed_groups = self._patchify(xyz)

        for m in self.SA_modules:
            xyz, features = m(xyz=xyz, features=features)

        if self.calc_loss:
            z = features.mean(dim=2)  # Average features of all points in each subcloud
            z = z.reshape(-1,
                          self.opt.subcloud_cube_size,
                          self.opt.subcloud_cube_size,
                          self.opt.subcloud_cube_size,
                          z.shape[1])
            z = z.permute(0,4,1,2,3)  # (B, C, cube_size, cube_size, cube_size)

            targets_to_ignore = failed_groups.reshape(-1,
                                                      self.opt.subcloud_cube_size,
                                                      self.opt.subcloud_cube_size,
                                                      self.opt.subcloud_cube_size)

            loss = self.loss(z, z, targets_to_ignore=targets_to_ignore)
        else:
            loss = None

        return xyz, features, loss, failed_groups
