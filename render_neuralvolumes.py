import time

import torch.utils.data
from torch.utils.data import DataLoader

from config_templates.blender2_config import DatasetConfig
from eval.NeuralVolumePlotter.NeuralVolumeBuilder import NeuralVolumeBuilder
from eval.NeuralVolumePlotter.NeuralVolumePlotter import NeuralVolumePlotter
from utils.EnvUtils import EnvUtils
from utils.ImportConfigUtil import ImportConfigUtil
from utils.RenderUtils import RenderUtils

if __name__ == "__main__":
    render_utils = RenderUtils()
    args = render_utils.parse_cmd_arguments()
    outpath = render_utils.get_outpath_and_print_infos(args)

    import_config_util = ImportConfigUtil()
    experconfig = import_config_util.import_module(args.experconfig)

    ds_config: DatasetConfig = experconfig.DatasetConfig()

    if args.traindataset:
        profile = ds_config.get_train_profile()
        mode = NeuralVolumeBuilder.MODE_TRAIN_DATASET
    else:
        profile = ds_config.get_render_profile()
        mode = NeuralVolumeBuilder.MODE_TEST_DATASET

    resolution: int = 64
    plotter = NeuralVolumePlotter(outpath, resolution, mode)

    env_utils = EnvUtils()
    batch_size = 1
    nof_workers = 8
    dataset = profile.get_dataset()
    ae = profile.get_autoencoder(dataset)
    torch.cuda.set_device(args.devices[0])
    ae = torch.nn.DataParallel(ae, device_ids=args.devices).to("cuda").eval()

    iterationPoint = args.iteration
    aeparams_file_name = "aeparams"
    if iterationPoint is not None:
        aeparams_file_name = "iteration{}_aeparams".format(iterationPoint)
    ae.module.load_state_dict(
        torch.load("{}/{}.pt".format(outpath, aeparams_file_name), map_location=torch.device('cuda', args.devices[0])),
        strict=False)

    dataloader_render = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=nof_workers)

    iternum = 0
    itemnum = 0
    starttime = time.time()

    imgindex = 0
    with torch.no_grad():
        for data in dataloader_render:
            b = next(iter(data.values())).size(0)
            # forward
            output = ae(iternum, [], **{k: x.to("cuda") for k, x in data.items()})
            decout_main = output['decout']

            plotter.save_volume_and_pos(decout=decout_main, frameidx=imgindex)

            endtime = time.time()
            ips = 1. / (endtime - starttime)
            print("{:4} / {:4} ({:.4f} iter/sec)".format(itemnum, len(dataset), ips), end="\n")
            starttime = time.time()
            iternum += 1
            itemnum += b
            imgindex += 1
            if imgindex >= 100:
                print("create plot for neural volumes")
                plotter.plot_frames()
                exit()
            else:
                print("saved neural volumes from frame with index {}".format(imgindex))
