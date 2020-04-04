from optparse import OptionGroup

def parse_GIM_args(parser):
    group = OptionGroup(parser, "Greedy InfoMax training options")
    group.add_option(
        "--learning_rate", type="float", default=2e-4, help="Learning rate"
    )
    group.add_option(
        "--subcloud_ball_radius",
        type="float",
        default=0.5,  # 0.5 with 3x3x3 cube means we miss ~0.2% of points during training
        help="Radius of ball used to collect points for each sub point cloud",
    )
    group.add_option(
        "--subcloud_num_points",
        type="int",
        default=256,
        help="Number of maximum points to be collected for each sub point cloud",
    )
    group.add_option(
        "--subcloud_cube_size",
        type="int",
        default=3,
        help="Size of cube to divide the each point cloud into. Size 2 = 2^3 = 8 sub clouds, size 3 = 3^3 = 27 sub clouds etc.",
    )
    group.add_option(
        "--negative_samples",
        type="int",
        default=8,
        help="Number of negative samples to be used for training",
    )
    group.add_option(
        "--model_splits",
        type="int",
        default=3,
        help="Number of individually trained modules that the original model should be split into "
             "options: 1 (normal end-to-end backprop) or 3 (default used in experiments of paper)",
    )
    group.add_option(
        "--train_module",
        type="int",
        default=3,
        help="Index of the module to be trained individually (0-2), "
             "or training network as one (3)",
    )

    parser.add_option_group(group)
    return parser
