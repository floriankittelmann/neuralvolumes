import argparse
import os
import sys
import time
import torch.utils.data
from train import import_module
from train import is_local_env
from torch.utils.data import DataLoader


def parse_arguments():
    parser = argparse.ArgumentParser(description='Render')
    parser.add_argument('experconfig', type=str, help='experiment config')
    parser.add_argument('--profile', type=str, default="Render", help='profile')
    parser.add_argument('--devices', type=int, nargs='+', default=[0], help='devices')
    parser.add_argument('--nofworkers', type=int, default=16, help='nofworkers')
    parser.add_argument('--cam', type=str, default="rotate", help='camera mode to render')
    args = parser.parse_args()
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg, type=eval)
    args = parser.parse_args()
    return args, parsed


def render(profile_local, args_local):
    nof_workers = args_local.nofworkers
    batch_size_training = profile_local.batchsize
    if is_local_env():
        nof_workers = 1
        batch_size_training = 3
    dataset = profile_local.get_dataset()
    ae = profile_local.get_autoencoder(dataset)
    torch.cuda.set_device(args_local.devices[0])
    ae = torch.nn.DataParallel(ae, device_ids=args_local.devices).to("cuda").eval()
    # load
    ae.module.load_state_dict(torch.load("{}/aeparams.pt".format(outpath), map_location=torch.device('cuda', args_local.devices[0])), strict=False)

    dataloader_render = DataLoader(dataset, batch_size=batch_size_training, shuffle=False, drop_last=True, num_workers=nof_workers)
    render_writer = profile_local.get_writer()

    iternum = 0
    itemnum = 0
    starttime = time.time()

    with torch.no_grad():
        for data in dataloader_render:
            b = next(iter(data.values())).size(0)
            # forward
            output = ae(iternum, [], **{k: x.to("cuda") for k, x in data.items()}, **profile_local.get_ae_args())

            render_writer.batch(iternum, itemnum + torch.arange(b), **data, **output)

            endtime = time.time()
            ips = 1. / (endtime - starttime)
            print("{:4} / {:4} ({:.4f} iter/sec)".format(itemnum, len(dataset), ips), end="\n")
            starttime = time.time()

            iternum += 1
            itemnum += b
    # cleanup
    render_writer.finalize()


if __name__ == "__main__":
    args_glob, parsed = parse_arguments()
    outpath = os.path.dirname(args_glob.experconfig)
    print(" ".join(sys.argv))
    print("Output path:", outpath)

    print("----- Evaluate on following devices -----")
    for device_id in args_glob.devices:
        print("GPU Device with ID {}".format(device_id))
        device = torch.cuda.get_device_properties(device_id)
        print("GPU Properties: {}, Memory: {} MB, ProzessorCount: {}".format(
            device.name,
            (device.total_memory / (2 * 1024)),
            device.multi_processor_count))

    experconfig = import_module(args_glob.experconfig, "config")
    profile_glob = getattr(experconfig, args_glob.profile)(**{k: v for k, v in vars(args_glob).items() if k not in parsed})
    # eval
    if args_glob.cam == "all":
        for camera_nr in range(36):
            camera = "{:03d}".format(camera_nr)
            print("start with camera " + camera)
            profile_glob.cam = camera
            render(profile_glob, args_glob)
    else:
        print(args_glob.cam)
        profile_glob.cam = args_glob.cam
        print(profile_glob.cam)
        render(profile_glob, args_glob)
