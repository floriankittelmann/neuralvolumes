from eval.CameraSetupPlotter import CameraSetupPlotter

if __name__ == "__main__":
    plotter = CameraSetupPlotter(CameraSetupPlotter.MODE_BLENDER2_DATASET)
    plotter.plot_camera_setup()
