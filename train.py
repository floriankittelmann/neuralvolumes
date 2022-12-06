# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Train an autoencoder."""
import sys
import time
import gc
import wandb
import numpy as np
import os
from utils.EnvUtils import EnvUtils
from utils.TrainUtils import TrainUtils
import torch.backends.cudnn

sys.dont_write_bytecode = True
torch.backends.cudnn.benchmark = True  # gotta go fast!

if __name__ == "__main__":
    train_utils = TrainUtils()
    args = train_utils.parse_cmd_arguments()
    config_path = train_utils.prepare_and_get_configpath(args)
    outpath, log = train_utils.get_outpath_and_print_infos(config_path, args)

    trainprofile, progressprof = train_utils.load_profiles(config_path)
    dataloader, testbatch, dataset = train_utils.build_datasets(trainprofile, progressprof, args)
    writer, ae, aeoptim, lossweights = train_utils.get_writer_autencoder_optimizer_lossweights(
        trainprofile,
        progressprof,
        dataset,
        args,
        outpath
    )


    # train
    starttime = time.time()
    #evalpoints = np.geomspace(1., trainprofile.get_maxiter(), 100).astype(np.int32)
    evalpoints = np.linspace(100, trainprofile.get_maxiter(), (trainprofile.get_maxiter() - 100) // 100)
    iternum = log.iternum
    prevloss = np.inf

    env_utils = EnvUtils()
    env = env_utils.get_env()
    epochs_to_learn = 10000

    run = wandb.init(
        project=env["wandb"]["project"],
        entity=env["wandb"]["entity"],
        name=os.path.basename(outpath),
        config={
            "experiment_path": outpath,
            "learning_rate": trainprofile.get_lr(),
            "epochs": epochs_to_learn,
            "batch_size": trainprofile.get_batchsize(),
            "mode": "offline"
        }
    )
    print("wandb sync " + os.path.dirname(run.dir))

    for epoch in range(epochs_to_learn):
        for data in dataloader:
            # forward
            output = ae(iternum, lossweights.keys(), **{k: x.to("cuda") for k, x in data.items()})

            # compute final loss
            loss = sum([
                lossweights[k] * (torch.sum(v[0]) / torch.sum(v[1]) if isinstance(v, tuple) else torch.mean(v))
                for k, v in output["losses"].items()])

            starttime = train_utils.print_iteration_infos(
                iternum=iternum,
                loss=loss,
                output=output,
                starttime=starttime)

            # update parameters
            aeoptim.zero_grad()
            loss.backward()
            aeoptim.step()

            # check for loss explosion
            if loss.item() > 20 * prevloss or not np.isfinite(loss.item()):
                print("Unstable loss function; resetting")

                ae.module.load_state_dict(torch.load("{}/aeparams.pt".format(outpath)), strict=False)
                aeoptim = trainprofile.get_optimizer(ae.module)

            prevloss = loss.item()

            np_img = None
            # save intermediate results
            if iternum % 1000 == 0 or iternum in [0, 1, 2, 3, 4, 5]:
                np_img = train_utils.save_model_and_validation_pictures(
                    iternum=iternum,
                    outpath=outpath,
                    ae=ae,
                    testbatch=testbatch,
                    progressprof=progressprof,
                    trainprofile=trainprofile,
                    data=data,
                    writer=writer)
            train_utils.save_wandb_info(iternum=iternum, loss=loss, output=output, img=np_img, wandb=wandb)

            iternum += 1
            torch.cuda.empty_cache()
            del loss
            del output
            gc.collect()

        if iternum >= trainprofile.get_maxiter():
            break

    # cleanup
    writer.finalize()
