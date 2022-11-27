import argparse
import os
import sys
import torch
import time
from utils.EnvUtils import EnvUtils
from torch.utils.data import DataLoader


class RenderUtils:

    def parse_cmd_arguments(self):
        parser = argparse.ArgumentParser(description='Render')
        parser.add_argument('experconfig', type=str, help='experiment config')
        parser.add_argument('--devices', type=int, nargs='+', default=[0], help='devices')
        parser.add_argument('--nofworkers', type=int, default=16, help='nofworkers')
        parser.add_argument('--cam', type=str, default="rotate", help='camera mode to render')
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
        # load
        ae.module.load_state_dict(
            torch.load("{}/aeparams.pt".format(outpath), map_location=torch.device('cuda', args.devices[0])),
            strict=False)

        dataloader_render = DataLoader(dataset, batch_size=batch_size_training, shuffle=False, drop_last=True,
                                       num_workers=nof_workers)
        render_writer = profile.get_writer()

        iternum = 0
        itemnum = 0
        starttime = time.time()

        with torch.no_grad():
            for data in dataloader_render:
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
