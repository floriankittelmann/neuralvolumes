import argparse
import os
import sys
import time

import torch.utils.data
from train import import_module
from train import is_local_env


def parse_arguments():
    parser = argparse.ArgumentParser(description='Render')
    parser.add_argument('experconfig', type=str, help='experiment config')
    parser.add_argument('--profile', type=str, default="Render", help='profile')
    parser.add_argument('--devices', type=int, nargs='+', default=[0], help='devices')
    parser.add_argument('--nofworkers', type=int, default=16, help='nofworkers')
    args = parser.parse_args()
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg, type=eval)
    args = parser.parse_args()
    return args, parsed


if __name__ == "__main__":
    args, parsed = parse_arguments()
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

    experconfig = import_module(args.experconfig, "config")
    profile = getattr(experconfig, args.profile)(**{k: v for k, v in vars(args).items() if k not in parsed})
    dataset = profile.get_dataset()
    nof_workers = args.nofworkers
    batch_size_training = profile.batchsize
    if is_local_env():
        nof_workers = 1
        batch_size_training = 3
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_training, shuffle=True, drop_last=True, num_workers=nof_workers)
    ae = profile.get_autoencoder(dataset)
    ae = torch.nn.DataParallel(ae, device_ids=args.devices).to("cuda").eval()

    writer = profile.get_writer()
    print("--- I am finished ----")

    # load
    ae.module.load_state_dict(torch.load("{}/aeparams.pt".format(outpath)), strict=False)

    # eval
    iternum = 0
    itemnum = 0
    starttime = time.time()

    with torch.no_grad():
        for data in dataloader:
            b = next(iter(data.values())).size(0)
            # forward
            output = ae(iternum, [], **{k: x.to("cuda") for k, x in data.items()}, **profile.get_ae_args())
            print(type(output["decout"]))

            break
            """
            writer.batch(iternum, itemnum + torch.arange(b), **data, **output)

            endtime = time.time()
            ips = 1. / (endtime - starttime)
            print("{:4} / {:4} ({:.4f} iter/sec)".format(itemnum, len(dataset), ips), end="\n")
            starttime = time.time()

            iternum += 1
            itemnum += b"""

    # cleanup
    #writer.finalize()

