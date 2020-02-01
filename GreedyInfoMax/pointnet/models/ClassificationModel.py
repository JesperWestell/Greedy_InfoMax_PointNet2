import torch.nn as nn

class ClassificationModel(nn.Sequential):
    def __init__(self, in_channels=1024, num_classes=40, hidden_nodes=[512, 256]):
        super(ClassificationModel, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        if not isinstance(hidden_nodes, list):
            hidden_nodes = [hidden_nodes]
        l = [in_channels] + hidden_nodes + [num_classes]

        for il in range(len(l) - 1):
            self.add_module("relu{}".format(il), nn.ReLU(inplace=True))
            self.add_module("fc_layer{}".format(il), nn.Linear(l[il], l[il + 1], bias=True))
