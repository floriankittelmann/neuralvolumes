import torch.utils.data
from config_templates.blender2_config import Train as TrainBlender
from config_templates.dryice1_config import Train as TrainDryice
from scipy.spatial.transform import Rotation as R
import numpy as np
import data.dryice1 as datamodel

if __name__ == "__main__":
    print(np.linspace(0.001, 0.5, 10))
    dataset = datamodel.Dataset(
        camerafilter=lambda x: True,
        framelist=[i for i in range(15469, 16578, 3)][:-1],
        keyfilter=["bg", "fixedcamimage", "camera", "image", "pixelcoords"],
        fixedcameras=["400007", "400010", "400018"],
        fixedcammean=100.,
        fixedcamstd=25.,
        imagemean=100.,
        imagestd=25.,
        subsamplesize=128,
        worldscale=1. / 256)
    """print("Blender")
    trainblender = TrainBlender()
    dataset = trainblender.get_dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True,
                                             num_workers=1)
    for data in dataloader:
        print(data["camindex"][0])
        print(data["campos"][0])
        print(data["camrot"][0])
        print(R.from_matrix(data["camrot"][0]).as_quat())
        break

    print("DryIce")
    traindrice = TrainDryice()
    dataset = traindrice.get_dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True,
                                             num_workers=1)
    for data in dataloader:
        print(data["camindex"][0])
        print(data["campos"][0])
        print(data["camrot"][0])
        print(R.from_matrix(data["camrot"][0]).as_quat())
        break"""

