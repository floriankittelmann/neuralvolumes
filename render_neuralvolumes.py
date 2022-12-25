import time

import torch.utils.data
from torch.utils.data import DataLoader
import os

from config_templates.blender2_config import DatasetConfig
from models.RayMarchingHelper import init_section_view
from models.volsamplers.warpvoxel import VolSampler
from utils.EnvUtils import EnvUtils
from utils.ImportConfigUtil import ImportConfigUtil
from utils.RenderUtils import RenderUtils
import numpy as np

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

    dataloader_render = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=nof_workers)
    #render_writer = profile.get_writer(outpath=outpath, nthreads=batch_size, is_plot_batch=True)
    iternum = 0
    itemnum = 0
    starttime = time.time()

    outpath_np_folder = os.path.join(outpath, "templates")
    try:
        os.makedirs(outpath_np_folder)
    except OSError:
        pass

    save_picture_from_side = False
    with torch.no_grad():
        for data in dataloader_render:
            b = next(iter(data.values())).size(0)
            # forward
            output = ae(iternum, [], **{k: x.to("cuda") for k, x in data.items()}, **profile.get_ae_args())

            np_filename = "frame{}.npy".format(iternum)
            path_np_file = os.path.join(outpath_np_folder, np_filename)
            template_tensor: torch.Tensor = output['decout']['template']
            template_np = template_tensor.cpu().numpy()
            with open(path_np_file, 'wb') as f:
                np.save(f, template_np)

            if save_picture_from_side:
                rayrgb = output["irgbrec"]
                raymarching_section_view = init_section_view(rayrgb.size(dim=0))

                section_rgb, section_alpha = raymarching_section_view.do_raymarching(
                    VolSampler(),
                    decout=output["decout"],
                    stepjitter=0.01,
                    viewtemplate=False
                )
                """
                render_writer.batch(
                    iternum,
                    itemnum + torch.arange(b),
                    irgbrec=section_rgb,
                    ialpharec=section_alpha,
                    image=None,
                    irgbsqerr=None
                )"""


            endtime = time.time()
            ips = 1. / (endtime - starttime)
            print("{:4} / {:4} ({:.4f} iter/sec)".format(itemnum, len(dataset), ips), end="\n")
            starttime = time.time()
            iternum += 1
            itemnum += b
            break
