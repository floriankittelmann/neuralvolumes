# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Train an autoencoder."""
import argparse
import importlib
import importlib.util
import sys
import time
import gc
import wandb
import numpy as np
import os
import torch.utils.data
import json
import shutil
from datetime import datetime
import re

sys.dont_write_bytecode = True
torch.backends.cudnn.benchmark = True  # gotta go fast!


class Logger(object):
    """Duplicates all stdout to a file."""

    def __init__(self, path, resume):
        if not resume and os.path.exists(path):
            print(path + " exists")
            sys.exit(0)

        iternum = 0
        if resume:
            with open(path, "r") as f:
                for line in f.readlines():
                    match = re.search("Iteration (\d+).* ", line)
                    if match is not None:
                        it = int(match.group(1))
                        if it > iternum:
                            iternum = it
        self.iternum = iternum

        self.log = open(path, "a") if resume else open(path, "w")
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, message):
        self.stdout.write(message)
        self.stdout.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def import_module(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_env() -> dict:
    file_name = 'env.json'
    env_dict = {}
    if os.path.exists(file_name):
        with open(file_name) as json_content:
            env_dict = json.load(json_content)
    return env_dict


def is_local_env() -> bool:
    return get_env()["env"] == "local"


def has_wandb() -> bool:
    return get_env()["env"] == "prod"


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Train an autoencoder')
    parser.add_argument('datasetname', type=str, nargs="?", default=None,
                        help='dataset name. a template config file is needed under config_templates and the data should be uploaded')
    parser.add_argument('experimentname', type=str, nargs="?", default=None, help='define an experiment name')
    parser.add_argument('--profile', type=str, default="Train", help='config profile')
    parser.add_argument('--devices', type=int, nargs='+', default=[0], help='devices')
    parser.add_argument('--resume', type=str, default=None, help='resume training and provide the config path')
    parser.add_argument('--nofworkers', type=int, default=16)
    parser.add_argument('--local', action='store_true',
                        help='training on local machine with small memory size and small gpu power')
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg, type=eval)
    args = parser.parse_args()

    if args.resume is None:
        if args.datasetname is None or args.experimentname is None:
            raise Exception(
                "When creating new training please provide the following arguments: datasetname, experimentname")
        dataset_name = args.datasetname
        templatefilename = dataset_name + "_config.py"
        path_template = os.path.join("config_templates", templatefilename)
        if not os.path.exists(path_template):
            raise Exception(path_template + " -> file does not exist ")
        print("found config template")

        root_experiment_path = os.path.join("experiments", dataset_name)
        if not os.path.exists(root_experiment_path) or not os.path.isdir(root_experiment_path):
            raise Exception(root_experiment_path + " -> directory does not exist")
        print("found experiments path")

        now = datetime.now()
        dt_string = now.strftime("%Y%m%d_%H%M%S")
        experiment_name = args.experimentname
        unique_experiment_name = dt_string + "_" + experiment_name
        experiment_path = os.path.join(root_experiment_path, unique_experiment_name)
        os.mkdir(experiment_path)
        print("created new experiment")

        config_path = os.path.join(experiment_path, "config.py")
        shutil.copyfile(path_template, config_path)
        print("copied config file")
    else:
        config_path = args.resume

    outpath = os.path.dirname(config_path)
    checkpoint_path = os.path.join(outpath, "checkpoint.tar")
    log = Logger("{}/log.txt".format(outpath), False if args.resume is None else True)
    print("Python", sys.version)
    print("PyTorch", torch.__version__)
    print(" ".join(sys.argv))
    print("Output path:", outpath)

    print("----- Train on following devices -----")
    for device_id in args.devices:
        print("GPU Device with ID {}".format(device_id))
        device = torch.cuda.get_device_properties(device_id)
        print("GPU Properties: {}, Memory: {} MB, ProzessorCount: {}".format(
            device.name,
            (device.total_memory / (2 * 1024)),
            device.multi_processor_count))

    # load config
    starttime = time.time()
    experconfig = import_module(config_path, "config")
    profile = getattr(experconfig, args.profile)(**{k: v for k, v in vars(args).items() if k not in parsed})
    progressprof = experconfig.Progress()
    print("Config loaded ({:.2f} s)".format(time.time() - starttime))

    # build dataset & testing dataset
    starttime = time.time()
    testdataset = progressprof.get_dataset()
    batch_size_test = progressprof.batchsize
    if is_local_env():
        batch_size_test = 3
    if len(testdataset) <= 0:
        raise Exception("problem appeared get_dataset")
    dataloader = torch.utils.data.DataLoader(testdataset, batch_size=batch_size_test, shuffle=False, drop_last=True,
                                             num_workers=0)
    if len(dataloader) <= 0:
        raise Exception("problem appeared dataloader")
    for testbatch in dataloader:
        break

    dataset = profile.get_dataset()
    nof_workers = args.nofworkers
    batch_size_training = profile.batchsize
    if is_local_env():
        nof_workers = 1
        batch_size_training = 3
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_training, shuffle=True, drop_last=True,
                                             num_workers=nof_workers)
    print("Dataset instantiated ({:.2f} s)".format(time.time() - starttime))

    # data writer
    starttime = time.time()
    writer = progressprof.get_writer()
    print("Writer instantiated ({:.2f} s)".format(time.time() - starttime))

    # build autoencoder
    starttime = time.time()
    ae = profile.get_autoencoder(dataset)
    torch.cuda.set_device(args.devices[0])
    ae = torch.nn.DataParallel(ae, device_ids=args.devices).to("cuda").train()
    if args.resume is not None:
        ae.module.load_state_dict(torch.load("{}/aeparams.pt".format(outpath)), strict=False)

    print("Autoencoder instantiated ({:.2f} s)".format(time.time() - starttime))

    # build optimizer
    starttime = time.time()
    aeoptim = profile.get_optimizer(ae.module)
    lossweights = profile.get_loss_weights()
    print("Optimizer instantiated ({:.2f} s)".format(time.time() - starttime))

    # train
    starttime = time.time()
    evalpoints = np.geomspace(1., profile.maxiter, 100).astype(np.int32)
    iternum = log.iternum
    prevloss = np.inf

    env = get_env()
    epochs_to_learn = 10000
    if has_wandb():
        wandb.init(
            project=env["wandb"]["project"],
            entity=env["wandb"]["entity"],
            resume=False if args.resume is None else True,
            name=os.path.basename(outpath),
            config={
                "experiment_path": outpath,
                "learning_rate": profile.lr,
                "epochs": epochs_to_learn,
                "batch_size": profile.batchsize
            }
        )
    #if args.resume is not None and has_wandb():
    #    dict_wandb = torch.load(wandb.restore(checkpoint_path))

    for epoch in range(epochs_to_learn):
        for data in dataloader:
            # forward
            output = ae(iternum, lossweights.keys(), **{k: x.to("cuda") for k, x in data.items()})

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

            if has_wandb():
                wandb.log(dict_wandb)
                wandb.watch(ae)

            # print current information
            print("Iteration {}: loss = {:.5f}, ".format(iternum, float(loss.item())) +
                  ", ".join(["{} = {:.5f}".format(k, float(
                      torch.sum(v[0]) / torch.sum(v[1]) if isinstance(v, tuple) else torch.mean(v)))
                             for k, v in output["losses"].items()]), end="")
            if iternum % 10 == 0:
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
                writer.batch(iternum, iternum * profile.batchsize + torch.arange(b), **testbatch, **testoutput)

            # update parameters
            aeoptim.zero_grad()
            loss.backward()
            aeoptim.step()

            # check for loss explosion
            if loss.item() > 20 * prevloss or not np.isfinite(loss.item()):
                print("Unstable loss function; resetting")

                ae.module.load_state_dict(torch.load("{}/aeparams.pt".format(outpath)), strict=False)
                aeoptim = profile.get_optimizer(ae.module)

            prevloss = loss.item()

            # save intermediate results
            if iternum % 5 == 0:
                torch.save(ae.module.state_dict(), "{}/aeparams.pt".format(outpath))
                if has_wandb():
                    torch.save(dict_wandb, checkpoint_path)
                    wandb.save(checkpoint_path)

            iternum += 1
            torch.cuda.empty_cache()
            del loss
            del output
            gc.collect()

        if iternum >= profile.maxiter:
            break

    # cleanup
    writer.finalize()
