import time
import os
import sys
import torch.utils.data
from train import import_module
from train import is_local_env
from torch.utils.data import DataLoader
from render_2 import parse_arguments

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
    ae = profile.get_autoencoder(dataset)
    torch.cuda.set_device(args.devices[0])
    ae = torch.nn.DataParallel(ae, device_ids=args.devices).to("cuda").eval()

    # load
    ae.module.load_state_dict(
        torch.load("{}/aeparams.pt".format(outpath), map_location=torch.device('cuda', args.devices[0])), strict=False)

    # eval
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
            print("output")
            print(output.keys())

            print("output -> irgbrec")
            print(type(output["irgbrec"]))

            print("output -> decout")
            print(type(output["decout"]))
            print(output["decout"].keys())
            exit()
