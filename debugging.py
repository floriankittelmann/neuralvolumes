import torch.utils.data
from experiments.blender2.experiment.config import Train as TrainBlender
from experiments.dryice1.experiment1.config import Train as TrainDryice

if __name__ == "__main__":
    print("Blender")
    trainblender = TrainBlender()
    dataset = trainblender.get_dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True,
                                             num_workers=1)
    for data in dataloader:
        print(data["campos"])

    print("DryIce")
    traindrice = TrainDryice()
    dataset = traindrice.get_dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True,
                                             num_workers=1)
    for data in dataloader:
        print(data["campos"])

