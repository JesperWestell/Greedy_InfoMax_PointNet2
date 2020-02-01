def parse_general_args(parser):
    parser.add_option(
        "--experiment",
        type="string",
        default="3D_pointclouds",
        help="not a real option, just for bookkeeping",
    )
    parser.add_option(
        "--dataset",
        type="string",
        default="modelnet40",
        help="Dataset to use for training, default: modelnet40",
    )
    parser.add_option(
        "--num_workers", type="int", default=8, help="Number of dataloader workers"
    )
    parser.add_option(
        "--num_points", type="int", default=1024, help="Number of points in each point cloud"
    )
    parser.add_option(
        "--num_unsupervised_training_samples", type="int", default=8000,
        help="Number of training examples to be used in unsupervised learning out of the total "
             "9843 in the ModelNet40 training data. The rest are used in supervised learning."
    )
    parser.add_option(
        "--num_epochs", type="int", default=300, help="Number of Epochs for Training"
    )
    parser.add_option("--seed", type="int", default=2, help="Random seed for training")
    parser.add_option("--batch_size", type="int", default=32, help="Batchsize")
    parser.add_option(
        "--resnet", type="int", default=50, help="Resnet version (options 34 and 50)"
    )
    parser.add_option(
        "-i",
        "--data_input_dir",
        type="string",
        default="./datasets",
        help="Directory to store bigger datafiles (dataset and models)",
    )
    parser.add_option(
        "-o",
        "--data_output_dir",
        type="string",
        default=".",
        help="Directory to store bigger datafiles (dataset and models)",
    )
    parser.add_option(
        "--validate",
        action="store_true",
        default=False,
        help="Boolean to decide whether to split train dataset into train/val and plot validation loss (True) or combine train+validation set for final testing (False)",
    )
    parser.add_option(
        "--loss",
        type="int",
        default=0,
        help="Loss function to use for training:"
        "0 - InfoNCE loss"
        "1 - supervised loss using class labels",
    )
    parser.add_option(
        "--grayscale",
        action="store_true",
        default=False,
        help="Boolean to decide whether to convert images to grayscale (default: true)",
    )
    parser.add_option(
        "--weight_init",
        action="store_true",
        default=False,
        help="Boolean to decide whether to use special weight initialization (delta orthogonal)",
    )
    parser.add_option(
        "--save_dir",
        type="string",
        default="",
        help="If given, uses this string to create directory to save results in "
        "(be careful, this can overwrite previous results); "
        "otherwise saves logs according to time-stamp",
    )
    return parser
