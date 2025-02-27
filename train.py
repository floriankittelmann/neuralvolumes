# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Train an autoencoder."""
import math
import sys
import time
import gc
import wandb
import numpy as np
import os

from eval.NeuralVolumePlotter.NeuralVolumeBuilder import NeuralVolumeBuilder
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
    train_dataloader, test_dataloader, dataset = train_utils.build_datasets(trainprofile, progressprof, args)
    writer, ae, aeoptim, lossweights = train_utils.get_writer_autencoder_optimizer_lossweights(
        trainprofile,
        progressprof,
        dataset,
        args,
        outpath
    )

    start_training_time = time.time()
    # train
    starttime = time.time()
    # evalpoints = np.geomspace(1., trainprofile.get_maxiter(), 100).astype(np.int32)
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

    train_with_ground_truth_loss = trainprofile.get_should_train_with_ground_truth()
    for epoch in range(epochs_to_learn):
        for data in train_dataloader:
            # forward
            output = ae(iternum, lossweights.keys(), **{k: x.to("cuda") for k, x in data.items()})

            ground_truth_loss_train = None
            if 'ground_truth_loss' in output.keys():
                ground_truth_loss_train = output['ground_truth_loss']
            train_loss = train_utils.calculate_final_loss_from_output(
                output=output,
                lossweights=lossweights,
                ground_turth_loss=ground_truth_loss_train,
                iternum=iternum,
                train_with_ground_truth=train_with_ground_truth_loss
            )

            starttime = train_utils.print_iteration_infos(
                iternum=iternum,
                loss=train_loss,
                output=output,
                starttime=starttime)

            # update parameters
            aeoptim.zero_grad()
            train_loss.backward()
            aeoptim.step()

            # check for loss explosion
            if train_loss.item() > 20 * prevloss or not np.isfinite(train_loss.item()):
                print("Unstable loss function; resetting")

                ae.module.load_state_dict(torch.load("{}/aeparams.pt".format(outpath)), strict=False)
                aeoptim = trainprofile.get_optimizer(ae.module)
            prevloss = train_loss.item()

            test_loss = None
            testoutput = None
            np_img = None
            test_batch = None
            ground_truth_loss_test = None
            # save intermediate results
            if (iternum < 100 and iternum % 20 == 0) or (100 <= iternum <= 1000 and iternum % 100 == 0) or iternum % 1000 == 0 or iternum in [0, 1, 2, 3, 4, 5]:
                test_batch, testoutput = train_utils.get_testbatch_testoutput(
                    iternum=iternum,
                    progressprof=progressprof,
                    test_dataloader=test_dataloader,
                    ae=ae,
                    lossweights=lossweights
                )

                ground_truth_loss_test = None
                if 'ground_truth_loss' in testoutput.keys():
                    ground_truth_loss_test = testoutput['ground_truth_loss']
                test_loss = train_utils.calculate_final_loss_from_output(
                    output=testoutput,
                    lossweights=lossweights,
                    ground_turth_loss=ground_truth_loss_test,
                    iternum=iternum,
                    train_with_ground_truth=train_with_ground_truth_loss
                )
                np_img = train_utils.save_model_and_validation_pictures(
                    iternum=iternum,
                    outpath=outpath,
                    ae=ae,
                    test_batch=test_batch,
                    testoutput=testoutput,
                    trainprofile=trainprofile,
                    data=data,
                    writer=writer)

            train_utils.save_wandb_info(
                iternum=iternum,
                train_loss=train_loss,
                train_output=output,
                test_loss=test_loss,
                test_output=testoutput,
                validation_img=np_img,
                ground_truth_loss_train=ground_truth_loss_train,
                ground_truth_loss_test=ground_truth_loss_test,
                wandb=wandb)

            iternum += 1
            torch.cuda.empty_cache()
            del train_loss
            del output
            if test_batch is not None:
                del test_batch
            if test_loss is not None:
                del test_loss
            if testoutput is not None:
                del testoutput
            gc.collect()

        if iternum >= trainprofile.get_maxiter():
            break

    torch.save(ae.module.state_dict(), "{}/last_aeparams.pt".format(outpath))
    def format_time_of_sec(time_needed_in_sec: float) -> str:
        time_needed_in_min = float(time_needed_in_sec) / 60.0
        time_needed_in_hours = time_needed_in_min / 60.0
        mins_formated = time_needed_in_min - int(math.floor(time_needed_in_hours)) * 60
        time_needed_in_days = time_needed_in_hours / 24.0
        days_formated = int(math.floor(time_needed_in_days))
        hours_formated = time_needed_in_hours - days_formated * 24
        return "{}days {}h {}min".format(days_formated, hours_formated, mins_formated)

    print("max iterations reached. finish training! -> Needed:  for {} iterations".format(format_time_of_sec(time.time() - start_training_time), iternum))
