# Using Greedy InfoMax with PointNet++
We use the Greedy InfoMax algorithm to train a PointNet++. Still in an experimentation phase. If successful, the algorithm may be used in learning deep representations of 3D point clouds using large unlabeled datasets, which can then be used in both classification and semantic segmentation.

## Dependencies and Usage
TBD

## Early Experiments
The 9843 models in the training dataset of ModelNet40 have been divided into unsupervised training data (9000) and supervised training data (843).

Greedy InfoMax have been performed on the unsupervised data, splitting each 1024-point pointcloud into 4x4x4 256-point subclouds of uniform spacing [-0.6, -0.2, 0.2, 0.6] in all 3 axes, using a sphere query of radius 0.4. The network have been trained unsupervised for 205 epochs using default hyperparameters.

The network has been fine-tuned using the training data, for 200 epochs, training only the fully-connected layers of the model.
For comparision, classification training has been performed in a fully supervised manner by training the whole network end-to-end, with randomly initialized parameters. 

Initial results show that with a very small amount of training data, we can indeed see a performance gain, if pre-training on unlabeled data.


## Acknowledgements
* [loeweX/Greedy_InfoMax](https://github.com/loeweX/Greedy_InfoMax): Paper author and official repo, Greedy Infomax.
* [charlesq34/pointnet2](https://github.com/charlesq34/pointnet2): Paper author and official code repo, PointNet++.
* [sshaoshuai/Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch): PyTorch implementation of PointNet++, mainly used in this repo.

## Papers
* [Putting An End to End-to-End: Gradient-Isolated Learning of Representations - Löwe et al.](https://arxiv.org/abs/1905.11786)
* [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space - Qi et al.](https://arxiv.org/abs/1706.02413)

