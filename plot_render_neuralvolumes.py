import time

import torch.utils.data
from torch.utils.data import DataLoader

from utils.EnvUtils import EnvUtils
from utils.ImportConfigUtil import ImportConfigUtil
from utils.RenderUtils import RenderUtils

if __name__ == "__main__":
    render_utils = RenderUtils()
    args = render_utils.parse_cmd_arguments()
    outpath = render_utils.get_outpath_and_print_infos(args)

    import_config_util = ImportConfigUtil()
    experconfig = import_config_util.import_module(args.experconfig)
    profile = experconfig.DatasetConfig().get_render_profile()

    env_utils = EnvUtils()
    nof_workers = args.nofworkers
    batch_size = profile.batchsize
    if env_utils.is_local_env():
        nof_workers = 1
        batch_size = 3
    dataset = profile.get_dataset()
    ae = profile.get_autoencoder(dataset)
    torch.cuda.set_device(args.devices[0])
    ae = torch.nn.DataParallel(ae, device_ids=args.devices).to("cuda").eval()

    # load
    ae.module.load_state_dict(
        torch.load("{}/aeparams.pt".format(outpath), map_location=torch.device('cuda', args.devices[0])),
        strict=False)

    dataloader_render = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=nof_workers)
    render_writer = profile.get_writer(outpath=outpath, nthreads=batch_size, is_plot_batch=True)
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
