import torch
import torch.nn as nn

from PointNet2.pointnet2.pointnet2_utils import QueryAndGroup

class PointCloudGrouper(nn.Module):
    """
    Splits a point cloud into multiple overlapping sub point clouds, to be used in the Greedy InfoMax algorithm
    """
    def __init__(self, opt):
        super(PointCloudGrouper, self).__init__()
        self.cube_size = opt.subcloud_cube_size
        self.centers = self._create_sub_cloud_centers(opt)
        self.splitter = QueryAndGroup(opt.subcloud_ball_radius, opt.subcloud_num_points)

    def _create_sub_cloud_centers(self, opt):
        centers = torch.Tensor([[x, y, z] for
                                  x in range(opt.subcloud_cube_size)
                                  for y in range(opt.subcloud_cube_size)
                                  for z in range(opt.subcloud_cube_size)]).unsqueeze(0).to(opt.device)
        centers -= (opt.subcloud_cube_size - 1) / 2
        centers *= opt.subcloud_ball_radius
        return centers

    def forward(self, xyz):
        # (B, N, 3)  ->  (B, 3, cube_size^3, num_points)
        xyz = self.splitter(xyz, self.centers)
        # (B, 3, cube_size^3, num_points)  ->  (B, cube_size^3, num_points, 3)
        xyz = xyz.permute(0, 2, 3, 1)
        # B, cube_size^3, num_points, 3)  ->  (B*cube_size^3, num_points, 3)
        xyz = xyz.reshape(xyz.shape[0]*xyz.shape[1], xyz.shape[2], xyz.shape[3])
        return xyz