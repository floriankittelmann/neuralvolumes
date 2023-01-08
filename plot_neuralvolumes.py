from eval.NeuralVolumePlotter.NeuralVolumePlotter import NeuralVolumePlotter
from utils.RenderUtils import RenderUtils
import os.path

if __name__ == "__main__":
    render_utils = RenderUtils()
    args = render_utils.parse_cmd_arguments()
    outpath = render_utils.get_outpath_and_print_infos(args)

    outpath_np_folder = os.path.join(outpath, "templates")

    if not os.path.exists(outpath_np_folder):
        print("the neural volumes needs to rendered. Please use render_neuralvolumes.py script")
        exit()

    plotter = NeuralVolumePlotter(outpath_np_folder)
    plotter.plot_frames()
