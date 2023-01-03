import argparse
import os
import sys
import torch
import time
import numpy as np
from utils.EnvUtils import EnvUtils
from torch.utils.data import DataLoader


def get_distributed_coords(batchsize: int, fixed_value: float, nof_points: int, fixed_axis: int) -> np.ndarray:
    if fixed_axis == 0:
        list_coordinates = [(fixed_value, y, z)
                            for y in np.linspace(-1.0, 1.0, nof_points)
                            for z in np.linspace(-1.0, 1.0, nof_points)
                            for i in range(batchsize)]
    elif fixed_axis == 1:
        list_coordinates = [(x, fixed_value, z)
                            for x in np.linspace(-1.0, 1.0, nof_points)
                            for z in np.linspace(-1.0, 1.0, nof_points)
                            for i in range(batchsize)]
    elif fixed_axis == 2:
        list_coordinates = [(x, y, fixed_value)
                            for x in np.linspace(-1.0, 1.0, nof_points)
                            for y in np.linspace(-1.0, 1.0, nof_points)
                            for i in range(batchsize)]
    else:
        raise Exception("parameter fixed axis should be the value of either 0, 1 or 2")
    start_coords = np.array(list_coordinates)
    return start_coords.reshape((batchsize, nof_points, nof_points, 3))


class RenderUtils:

    def parse_cmd_arguments(self):
        parser = argparse.ArgumentParser(description='Render')
        parser.add_argument('experconfig', type=str, help='experiment config')
        parser.add_argument('--devices', type=int, nargs='+', default=[0], help='devices')
        parser.add_argument('--nofworkers', type=int, default=16, help='nofworkers')
        parser.add_argument('--cam', type=str, default="rotate", help='camera mode to render')
        parser.add_argument('--iteration', type=int, default=None, help='iteration of aeparams to choose')
        parser.add_argument('--startframe', type=int, default=None, help='start rendering from frame')
        parser.add_argument('--endframe', type=int, default=None, help='end rendering frame')
        parser.add_argument('--traindataset', action='store_true', help='use train dataset to render instead test dataset')
        parsed, unknown = parser.parse_known_args()
        for arg in unknown:
            if arg.startswith(("-", "--")):
                parser.add_argument(arg, type=eval)
        return parser.parse_args()

    def get_outpath_and_print_infos(self, args):
        outpath = os.path.dirname(args.experconfig)
        print(" ".join(sys.argv))
        print("Output path:", outpath)

        print("----- Evaluate on following devices -----")
        for device_id in args.devices:
            print("GPU Device with ID {}".format(device_id))
            device = torch.cuda.get_device_properties(device_id)
            print("GPU Properties: {}, Memory: {} MB, ProzessorCount: {}".format(
                device.name,
                (device.total_memory / (2 * 1024)),
                device.multi_processor_count))
        return outpath

    def render(self, profile, args, outpath):
        env_utils = EnvUtils()
        nof_workers = args.nofworkers
        batch_size_training = profile.batchsize
        if env_utils.is_local_env():
            nof_workers = 1
            batch_size_training = 3
        dataset = profile.get_dataset()
        ae = profile.get_autoencoder(dataset)
        torch.cuda.set_device(args.devices[0])
        ae = torch.nn.DataParallel(ae, device_ids=args.devices).to("cuda").eval()

        iterationPoint = args.iteration
        aeparams_file_name = "aeparams"
        if iterationPoint is not None:
            aeparams_file_name = "iteration{}_aeparams".format(iterationPoint)
        ae.module.load_state_dict(
            torch.load("{}/{}.pt".format(outpath, aeparams_file_name),
                       map_location=torch.device('cuda', args.devices[0])),
            strict=False)

        dataloader_render = DataLoader(dataset, batch_size=batch_size_training, shuffle=False, drop_last=True,
                                       num_workers=nof_workers)
        render_writer = profile.get_writer(outpath, batch_size_training)

        iternum = 0
        itemnum = 0
        starttime = time.time()

        with torch.no_grad():
            for data in dataloader_render:
                #print(data)
                #exit()
                b = next(iter(data.values())).size(0)
                # forward
                output = ae(iternum, [], **{k: x.to("cuda") for k, x in data.items()}, **profile.get_ae_args())

                render_writer.batch(iternum, itemnum + torch.arange(b), **data, **output)

                endtime = time.time()
                ips = 1. / (endtime - starttime)
                print("{:4} / {:4} ({:.4f} iter/sec)".format(itemnum, len(dataset), ips), end="\n")
                starttime = time.time()

                iternum += 1
                itemnum += b
        # cleanup
        render_writer.finalize()
