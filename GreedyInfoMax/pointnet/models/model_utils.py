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
        centers = centers.repeat(opt.batch_size, 1, 1)
        return centers

    def forward(self, xyz):
        # (B, N, 3)  ->  (B, 3, cube_size^3, num_points)
        xyz = self.splitter(xyz, self.centers)
        # (B, 3, cube_size^3, num_points)  ->  (B, cube_size^3, num_points, 3)
        xyz = xyz.permute(0, 2, 3, 1)
        # B, cube_size^3, num_points, 3)  ->  (B*cube_size^3, num_points, 3)
        xyz = xyz.reshape(xyz.shape[0]*xyz.shape[1], xyz.shape[2], xyz.shape[3])

        # If there are no points to gather within the sphere for a group, the function used in QueryAndGroup will
        # simply fill the group with copies of the center point. While this works without problems when grouping in
        # the PointNet++ algorithm, in our case it will result in creating points that don't exist in the original
        # point cloud. To fix this, we need to keep track of these 'failed' groups so that we ignore them in the
        # loss function.
        failed_groups = torch.eq(torch.eq(xyz[:,1], xyz[:,2]).sum(dim=1),3)

        return xyz, failed_groups