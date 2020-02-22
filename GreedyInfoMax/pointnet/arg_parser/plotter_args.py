def parse_plotter_args(parser):
    parser.add_option(
        "--name_of_3dmodel",
        type="string",
        default="airplane/airplane_0630.ply",
        help="name of 3D model to plot using the 'plot_subclouds' script",
    )
    parser.add_option(
        "--save_plot_frames",
        action="store_true",
        default=False,
        help="whether to record the 3D model shown in 'plot_subclouds' script",
    )
    parser.add_option(
        "--plotted_image_folder",
        type="string",
        default="./gif_images",
        help="folder to store the images created if save_plot_frames=True",
    )

    return parser
