import time

import torch.utils.data
from torch.utils.data import DataLoader
import os

from config_templates.blender2_config import DatasetConfig
from models.RayMarchingHelper import init_section_view, init_with_camera_position, RayMarchingHelper
from models.volsamplers.warpvoxel import VolSampler
from utils.EnvUtils import EnvUtils
from utils.ImportConfigUtil import ImportConfigUtil
from utils.RenderUtils import RenderUtils
import numpy as np


def get_uniform_positions(decout: dict) -> torch.Tensor:
    template: torch.Tensor = decout['template']
    template: np.ndarray = template.cpu().numpy()
    density: float = 16.0
    min: float = -1.0
    max: float = 1.0
    distribution: np.ndarray = np.arange(min, max, (2.0 / density))
    x, y, z = np.meshgrid(distribution, distribution, distribution)
    pos = np.stack((x, y, z), axis=3)
    dimension = int(density ** 3)
    batchsize = template.shape[0]
    pos = np.array([pos for i in range(batchsize)])
    pos = pos.reshape((batchsize, 1, dimension, 1, 3))
    torch.cuda.set_device("cuda:0")
    cur_device = torch.cuda.current_device()
    pos = torch.from_numpy(pos)
    return pos.to(cur_device)


def save_uniform_dist_volume(decout: dict):
    pos: torch.Tensor = get_uniform_positions(decout)
    save_volume_and_pos(pos, 'volume.npy', 'position.npy')


def save_volume_and_pos(pos: torch.Tensor, volume_name: str, pos_name: str):
    volsampler: VolSampler = VolSampler()
    sample_rgb, sample_alpha = volsampler(pos=pos, **decout)
    sample_rgb: np.ndarray = sample_rgb.cpu().numpy()
    sample_alpha: np.ndarray = sample_alpha.cpu().numpy()
    pos: np.ndarray = pos.cpu().numpy()
    shape: tuple = sample_rgb.shape
    nof_data_points = shape[3]

    batchsize = shape[0]
    sample_rgba: np.ndarray = np.zeros((batchsize, nof_data_points, 4))

    sample_rgb = sample_rgb.reshape((batchsize, nof_data_points, 3))
    sample_alpha = sample_alpha.reshape((batchsize, nof_data_points))
    sample_rgba[:, :, 0:3] = sample_rgb
    sample_rgba[:, :, 3] = sample_alpha
    pos = pos.reshape((batchsize, nof_data_points, 3))

    with open(volume_name, 'wb') as f:
        np.save(f, sample_rgba)

    with open(pos_name, 'wb') as f:
        np.save(f, pos)


def save_volume_from_camera(data: dict, decout: dict, dt: float):
    pixelcoords = data['pixelcoords'].to('cuda')
    princpt = data['princpt'].to('cuda')
    camrot = data['camrot'].to('cuda')
    focal = data['focal'].to('cuda')
    campos = data['campos'].to('cuda')
    raymarchHelper: RayMarchingHelper = init_with_camera_position(
        pixelcoords=pixelcoords,
        princpt=princpt,
        camrot=camrot,
        focal=focal,
        campos=campos,
        dt=dt)
    raypos_appended = None
    for raypos in raymarchHelper.iterate_raypos():
        raypos = raypos.cpu().numpy()
        shape = raypos.shape
        batchsize = shape[0]
        dimension = shape[1] * shape[2]
        raypos = raypos.reshape((batchsize, 1, dimension, 1, 3))
        if raypos_appended is None:
            raypos_appended = raypos
        else:
            raypos_appended = np.append(raypos_appended, raypos, axis=2)
    pos: torch.Tensor = torch.from_numpy(raypos_appended)
    save_volume_and_pos(pos, 'volume_cam.npy', 'position_cam.npy')


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

    dataloader_render = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True,
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

    should_save_cam_volume = True
    with torch.no_grad():
        for data in dataloader_render:
            b = next(iter(data.values())).size(0)
            # forward
            output = ae(iternum, [], **{k: x.to("cuda") for k, x in data.items()}, **profile.get_ae_args())
            decout = output['decout']

            if should_save_cam_volume:
                save_volume_from_camera(data=data, decout=decout, dt=(2. / float(32)))
            else:
                save_uniform_dist_volume(decout=decout)

            endtime = time.time()
            ips = 1. / (endtime - starttime)
            print("{:4} / {:4} ({:.4f} iter/sec)".format(itemnum, len(dataset), ips), end="\n")
            starttime = time.time()
            iternum += 1
            itemnum += b
            break
