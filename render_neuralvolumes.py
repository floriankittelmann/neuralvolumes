import time

import torch.utils.data
from torch.utils.data import DataLoader
import os

from config_templates.blender2_config import DatasetConfig
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
    profile = ds_config.get_render_profile()

    env_utils = EnvUtils()
    nof_workers = args.nofworkers
    batch_size = profile.batchsize
    if env_utils.is_local_env():
        nof_workers = 1
        batch_size = 3
    batch_size = 1
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
    # render_writer = profile.get_writer(outpath=outpath, nthreads=batch_size, is_plot_batch=True)
    iternum = 0
    itemnum = 0
    starttime = time.time()

    outpath_np_folder = os.path.join(outpath, "templates")
    try:
        os.makedirs(outpath_np_folder)
    except OSError:
        pass

    plotter = NeuralVolumePlotter(outpath_np_folder)
    imgindex = 0
    with torch.no_grad():
        for data in dataloader_render:
            b = next(iter(data.values())).size(0)
            # forward
            output = ae(iternum, [], **{k: x.to("cuda") for k, x in data.items()}, **profile.get_ae_args())
            decout_main = output['decout']

            plotter.save_uniform_dist_volume(decout=decout_main, frameidx=imgindex)

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
