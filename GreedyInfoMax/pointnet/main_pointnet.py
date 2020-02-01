import torch
import time
import numpy as np

#### own modules
from GreedyInfoMax.utils import logger
from GreedyInfoMax.pointnet.arg_parser import arg_parser
#from GreedyInfoMax.vision.models import load_vision_model
from GreedyInfoMax.pointnet.data import get_dataloader












if __name__ == "__main__":
    opt = arg_parser.parse_args()
    print(opt)
    opt.data_input_dir ='/home/jesper/git-repos/Greedy_InfoMax/datasets'

    # random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    if opt.device.type != "cpu":
        torch.backends.cudnn.benchmark = True

    # for now
    opt.batch_size_multiGPU = opt.batch_size

    train_loader, _, supervised_loader, _, test_loader, _ = get_dataloader.get_dataloader(
        opt
    )
    print('done')