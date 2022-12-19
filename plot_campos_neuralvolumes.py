from eval.CameraSetupPlotter import CameraSetupPlotter
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Render')
    parser.add_argument('--renderrot', action='store_true')
    args = parser.parse_args()
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg, type=eval)
    args = parser.parse_args()
    if args.renderrot:
        plotter = CameraSetupPlotter(CameraSetupPlotter.MODE_ROT_RENDER)
        plotter.plot_rotrender()
    else:
        plotter = CameraSetupPlotter(CameraSetupPlotter.MODE_BLENDER2_DATASET)
        plotter.plot_camera_setup()
