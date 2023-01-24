import time

import torch.utils.data
from torch.utils.data import DataLoader

from config_templates.blender2_config import DatasetConfig
from eval.NeuralVolumePlotter.GroundTruthLoss import GroundTruthLoss
from eval.NeuralVolumePlotter.NeuralVolumeBuilder import NeuralVolumeBuilder
from eval.NeuralVolumePlotter.NeuralVolumePlotter import NeuralVolumePlotter
from utils.EnvUtils import EnvUtils
from utils.ImportConfigUtil import ImportConfigUtil
from utils.RenderUtils import RenderUtils
from utils.TrainUtils import TrainUtils
import numpy as np

if __name__ == "__main__":
    render_utils = RenderUtils()
    args = render_utils.parse_cmd_arguments()
    outpath = render_utils.get_outpath_and_print_infos(args)

    import_config_util = ImportConfigUtil()
    experconfig = import_config_util.import_module(args.experconfig)

    ds_config: DatasetConfig = experconfig.DatasetConfig()
    if args.traindataset:
        profile = ds_config.get_train_profile()
    else:
        profile = ds_config.get_render_profile()

    resolution: int = 16
    plotter = NeuralVolumePlotter(resolution)

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

    nof_tests = 100
    train_utils = TrainUtils()
    train_profile = ds_config.get_train_profile()
    lossweights = train_profile.get_loss_weights()

    listGt: list = []
    listTestLoss: list = []

    imgindex = 0
    with torch.no_grad():
        for data in dataloader_render:
            b = next(iter(data.values())).size(0)
            # forward
            output = ae(iternum, lossweights, **{k: x.to("cuda") for k, x in data.items()})

            if iternum == 0:
                plotter.plot_one_frame(decout=output['decout'], input=data)
            
            exit()
            ground_positions = data["gt_positions"].to("cuda")
            ground_volume = data["gt_volume"].to("cuda")
            grountTruthLoss = GroundTruthLoss(
                decout=output['decout'],
                pos_truth=ground_positions,
                volume_truth=ground_volume,
                resolution=resolution
            )

            listGt.append(grountTruthLoss.calculate_mse_loss().item())
            listTestLoss.append(train_utils.calculate_final_loss_from_output(
                output=output,
                lossweights=lossweights).item())
            iternum += 1
            if nof_tests <= iternum:
                listGt: np.ndarray = np.asarray(listGt)
                listTestLoss: np.ndarray = np.asarray(listTestLoss)
                print("-------- ground truth -------")
                print("shape")
                print(listGt.shape)
                print("mean")
                print(np.mean(listGt))
                print("std")
                print(np.std(listGt))
                print("")

                print("-------- test loss -------")
                print("shape")
                print(listTestLoss.shape)
                print("mean")
                print(np.mean(listTestLoss))
                print("std")
                print(np.std(listTestLoss))
                print("")
                exit()

            """endtime = time.time()
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
                print("saved neural volumes from frame with index {}".format(imgindex))"""
