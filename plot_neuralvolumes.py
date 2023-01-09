from eval.NeuralVolumePlotter.NeuralVolumeBuilder import NeuralVolumeBuilder
from eval.NeuralVolumePlotter.NeuralVolumePlotter import NeuralVolumePlotter
from utils.RenderUtils import RenderUtils
import os.path

if __name__ == "__main__":
    render_utils = RenderUtils()
    args = render_utils.parse_cmd_arguments()
    outpath = render_utils.get_outpath_and_print_infos(args)

    if args.traindataset:
        mode = NeuralVolumeBuilder.MODE_TRAIN_DATASET
    else:
        mode = NeuralVolumeBuilder.MODE_TEST_DATASET

    resolution: int = 64
    plotter = NeuralVolumePlotter(outpath, resolution, mode)
    plotter.plot_frames()
