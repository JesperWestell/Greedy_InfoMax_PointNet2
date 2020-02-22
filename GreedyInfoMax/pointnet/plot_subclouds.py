import torch
import numpy as np

## own modules
from GreedyInfoMax.pointnet.data import get_dataloader
from GreedyInfoMax.pointnet.arg_parser import arg_parser
from GreedyInfoMax.pointnet.models import load_pointnet_model

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import pptk

def rotate_3dcloud(cloud):
    """
    Rotates the given 3d point cloud 90 degrees around the x axis

    Arguments:
    cloud: NumpY Array of size (N,3)
    """
    theta = np.radians(90)
    c, s = np.cos(theta), np.sin(theta)
    Rx = np.array(((1,0,0),(0, c, -s), (0, s, c)))
    return cloud.dot(Rx.T)

def extract_subclouds(opt, context_model, test_loader):
    context_model.eval()

    idx = 12

    for step, (img, target) in enumerate(test_loader):

        model_input = img.to(opt.device)

        full_cloud = model_input[idx].cpu().data.numpy()
        print("full cloud:", full_cloud.shape)

        centers = context_model.module.encoder[0].grouper.centers[0].cpu().data.numpy()
        print(centers.shape)
        print(centers)

        full_cloud = rotate_3dcloud(full_cloud)
        centers = rotate_3dcloud(centers)

        points = np.concatenate([full_cloud, centers], axis=0)
        colors = np.concatenate([np.repeat(np.asarray(((1,1,1),)),1024, axis=0),
                                        np.repeat(np.asarray(((1,0.5,0.5),)),centers.shape[0], axis=0)], axis=0)

        print(points.shape)
        print(colors.shape)

        v = pptk.viewer(points, colors)
        v.set(point_size=0.01)


        with torch.no_grad():
            print("input shape:", model_input.shape)
            subclouds, failed_groups = context_model.module.encoder[0]._patchify(model_input)
            print("subclouds shape:", subclouds.shape)

            subclouds = subclouds.reshape(-1,
                          opt.subcloud_cube_size,
                          opt.subcloud_cube_size,
                          opt.subcloud_cube_size,
                          subclouds.shape[1],
                          subclouds.shape[2])
            print("subclouds reshape shape:", subclouds.shape)

            failed_groups = failed_groups.reshape(-1,
                                  opt.subcloud_cube_size,
                                  opt.subcloud_cube_size,
                                  opt.subcloud_cube_size)

        all_points = np.zeros((opt.subcloud_cube_size**3*opt.subcloud_num_points,3))

        for x in range(opt.subcloud_cube_size):
            for y in range(opt.subcloud_cube_size):
                for z in range(opt.subcloud_cube_size):

                    if failed_groups[idx,x,y,z]:
                        continue

                    i = x*opt.subcloud_cube_size**2 + y*opt.subcloud_cube_size + z

                    subcloud = subclouds[idx,x,y,z].cpu().data.numpy()
                    subcloud = rotate_3dcloud(subcloud)

                    c = centers[i][np.newaxis]

                    subcloud += 3*c

                    all_points[i*opt.subcloud_num_points: i*opt.subcloud_num_points+opt.subcloud_num_points] = subcloud

        v = pptk.viewer(all_points)
        v.set(point_size=0.01)

        return




if __name__ == "__main__":

    opt = arg_parser.parse_args()

    opt.subcloud_ball_radius = 0.4
    opt.subcloud_cube_size = 4
    opt.subcloud_num_points = 256

    opt.batch_size = 16
    opt.data_input_dir = "/home/jesper/git-repos/Greedy_InfoMax_with_PointNet/datasets"

    print(opt)

    # random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # load context model
    reload_model = True if opt.loss == "classifier" else False
    context_model, _ = load_pointnet_model.load_model_and_optimizer(opt, reload_model=reload_model)

    _, _, _, _, test_loader, _ = get_dataloader.get_dataloader(opt)


    patches = extract_subclouds(opt, context_model, test_loader)
