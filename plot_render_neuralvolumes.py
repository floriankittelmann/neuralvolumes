import time
import os
import sys
import torch.utils.data
from pymeshfix_test import plot_nv_from_decout
from train import import_module
from train import is_local_env
from torch.utils.data import DataLoader
from render import parse_arguments
import numpy as np


def save_template_as_np_array(template: torch.Tensor):
    np_template = template.cpu().numpy()
    volume_t0 = np_template[0, :, :, :, :]
    with open('test.npy', 'wb') as f:
        np.save(f, volume_t0)


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
    dataset = profile.get_dataset_config_func()
    nof_workers = args.nofworkers
    batch_size_training = profile.batchsize
    if is_local_env():
        nof_workers = 1
        batch_size_training = 4
    ae = profile.get_autoencoder_config_func(dataset)
    torch.cuda.set_device(args.devices[0])
    ae = torch.nn.DataParallel(ae, device_ids=args.devices).to("cuda").eval()

    # load
    ae.module.load_state_dict(
        torch.load("{}/aeparams.pt".format(outpath), map_location=torch.device('cuda', args.devices[0])), strict=False)

    # eval
    dataloader_render = DataLoader(dataset, batch_size=batch_size_training, shuffle=False, drop_last=True,
                                   num_workers=nof_workers)

    iternum = 0
    itemnum = 0
    starttime = time.time()

    with torch.no_grad():
        for data in dataloader_render:
            b = next(iter(data.values())).size(0)
            # forward
            output = ae(iternum, [], **{k: x.to("cuda") for k, x in data.items()}, **profile.get_ae_args())

            plot_nv_from_decout(output["decout"])
            print("worked")
            exit()
            template_tensor = output["decout"]["template"]
            save_template_as_np_array(template_tensor)
            iternum += 1
            itemnum += b
