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
    evalpoints = np.geomspace(1., trainprofile.get_maxiter(), 100).astype(np.int32)
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
            start_time = time.time()
            # forward
            output = ae(iternum, lossweights.keys(), **{k: x.to("cuda") for k, x in data.items()})
            print("needed time for forward: {}".format(time.time() - start_time))

            start_time = time.time()
            # compute final loss
            loss = sum([
                lossweights[k] * (torch.sum(v[0]) / torch.sum(v[1]) if isinstance(v, tuple) else torch.mean(v))
                for k, v in output["losses"].items()])

            dict_wandb = {"loss": float(loss.item()), "step": iternum}
            for k, v in output["losses"].items():
                if isinstance(v, tuple):
                    dict_wandb[k] = float(torch.sum(v[0]) / torch.sum(v[1]))
                else:
                    dict_wandb[k] = float(torch.mean(v))
            wandb.log(dict_wandb)

            # print current information
            print("Iteration {}: loss = {:.5f}, ".format(iternum, float(loss.item())) +
                  ", ".join(["{} = {:.5f}".format(k, float(
                      torch.sum(v[0]) / torch.sum(v[1]) if isinstance(v, tuple) else torch.mean(v)))
                             for k, v in output["losses"].items()]), end="")
            if iternum % 5 == 0:
                endtime = time.time()
                ips = 10. / (endtime - starttime)
                print(", iter/sec = {:.2f}".format(ips))
                starttime = time.time()
            else:
                print()

            # compute evaluation output
            if iternum in evalpoints:
                with torch.no_grad():
                    testoutput = ae(iternum, [], **{k: x.to("cuda") for k, x in testbatch.items()},
                                    **progressprof.get_ae_args())

                b = data["campos"].size(0)
                writer.batch(iternum, iternum * trainprofile.get_batchsize() + torch.arange(b), outpath, **testbatch, **testoutput)
            print("needed time for loss calc: {}".format(time.time() - start_time))

            start_time = time.time()
            # update parameters
            aeoptim.zero_grad()
            print("needed time for grad calc: {}".format(time.time() - start_time))
            start_time = time.time()
            loss.backward()
            print("needed time for backprop: {}".format(time.time() - start_time))
            start_time = time.time()
            aeoptim.step()
            print("needed time for optimiser step: {}".format(time.time() - start_time))

            # check for loss explosion
            if loss.item() > 20 * prevloss or not np.isfinite(loss.item()):
                print("Unstable loss function; resetting")

                ae.module.load_state_dict(torch.load("{}/aeparams.pt".format(outpath)), strict=False)
                aeoptim = trainprofile.get_optimizer(ae.module)

            prevloss = loss.item()

            # save intermediate results
            if iternum % 1000 == 0:
                torch.save(ae.module.state_dict(), "{}/aeparams.pt".format(outpath))

            iternum += 1
            torch.cuda.empty_cache()
            del loss
            del output
            gc.collect()
            if iternum >= 55:
                exit()

        if iternum >= trainprofile.get_maxiter():
            break

    # cleanup
    writer.finalize()
