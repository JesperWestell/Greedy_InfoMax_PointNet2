import torch
import pptk
import numpy as np

## own modules
from GreedyInfoMax.pointnet.data import get_dataloader
from GreedyInfoMax.pointnet.arg_parser import arg_parser
from GreedyInfoMax.pointnet.models import load_pointnet_model

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

def get_rotating_poses(angle=np.pi/4, radius=3):
    poses = []
    poses.append([0, 0, 0, 0 * np.pi / 2, angle, radius])
    poses.append([0, 0, 0, 1 * np.pi / 2, angle, radius])
    poses.append([0, 0, 0, 2 * np.pi / 2, angle, radius])
    poses.append([0, 0, 0, 3 * np.pi / 2, angle, radius])
    poses.append([0, 0, 0, 4 * np.pi / 2, angle, radius])
    return poses

def show_subclouds(opt, context_model, test_loader):
    context_model.eval()

    idx = 0

    img, _ = test_loader.dataset.__getitem__(test_loader.dataset.names.index(opt.name_of_3dmodel))

    model_input = torch.tensor(img, device=opt.device).unsqueeze(0)

    full_cloud = model_input[idx].cpu().data.numpy()

    centers = context_model.module.encoder[0].grouper.centers[0].cpu().data.numpy()

    full_cloud = rotate_3dcloud(full_cloud)
    centers = rotate_3dcloud(centers)

    points = np.concatenate([full_cloud, centers], axis=0)
    colors = np.concatenate([np.repeat(np.asarray(((1,1,1),)),1024, axis=0),
                                    np.repeat(np.asarray(((1,0.5,0.5),)),centers.shape[0], axis=0)], axis=0)

    v1 = pptk.viewer(points, colors)


    with torch.no_grad():
        subclouds, failed_groups = context_model.module.encoder[0]._patchify(model_input)

        subclouds = subclouds.reshape(-1,
                      opt.subcloud_cube_size,
                      opt.subcloud_cube_size,
                      opt.subcloud_cube_size,
                      subclouds.shape[1],
                      subclouds.shape[2])

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

    v2 = pptk.viewer(all_points)

    v1.set(point_size=0.01, show_info=False, show_axis=False)
    v2.set(point_size=0.01, show_info=False, show_axis=False)


    if not opt.save_plot_frames:
        v1.play(get_rotating_poses(angle=np.pi/4, radius=3),
                4 * np.arange(5), repeat=True, interp='linear')
        v2.play(get_rotating_poses(angle=np.pi/4, radius=7),
                4 * np.arange(5), repeat=True, interp='linear')
    else:
        try:
            import os
            os.mkdir(opt.plotted_image_folder)
        except:
            pass
        v1_name = opt.name_of_3dmodel.split("/")[1][:-4] + "_complete"
        v2_name = opt.name_of_3dmodel.split("/")[1][:-4] + "_patches"
        v1.record(opt.plotted_image_folder,
                  get_rotating_poses(angle=np.pi / 4, radius=3),
                2 * np.arange(5), interp='linear', prefix=v1_name)
        v2.record(opt.plotted_image_folder,
                  get_rotating_poses(angle=np.pi / 4, radius=7),
                2 * np.arange(5), interp='linear', prefix=v2_name)


if __name__ == "__main__":

    opt = arg_parser.parse_args()

    opt.batch_size = 1
    opt.num_unsupervised_training_samples = 1

    #opt.data_input_dir = "/home/jesper/git-repos/Greedy_InfoMax_with_PointNet/datasets"

    opt.save_plot_frames = False
    #opt.plotted_image_folder = "/home/jesper/git-repos/Greedy_InfoMax_with_PointNet/gif_images"

    #opt.name_of_3dmodel = "airplane/airplane_0630.ply"
    #opt.name_of_3dmodel = "toilet/toilet_0387.ply"

    # random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # load context model
    reload_model = True if opt.loss == "classifier" else False
    context_model, _ = load_pointnet_model.load_model_and_optimizer(opt, reload_model=reload_model)

    _, _, _, _, test_loader, _ = get_dataloader.get_dataloader(opt)

    print("3D Models: \n" + ", ".join(test_loader.dataset.names))

    patches = show_subclouds(opt, context_model, test_loader)
