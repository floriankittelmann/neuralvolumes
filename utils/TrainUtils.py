import argparse
import os
import shutil
from datetime import datetime
import time

import numpy as np

from utils.ImportConfigUtil import ImportConfigUtil
from utils.EnvUtils import EnvUtils
import torch
import torch.utils.data
import sys
from utils.Logger import Logger


class TrainUtils:

    def parse_cmd_arguments(self):
        # parse arguments
        parser = argparse.ArgumentParser(description='Train an autoencoder')
        parser.add_argument('datasetname', type=str, nargs="?", default=None,
                            help='dataset name. a template config file '
                                 'is needed under config_templates and '
                                 'the data '
                                 'should be uploaded')
        parser.add_argument('experimentname', type=str, nargs="?", default=None, help='define an experiment name')
        parser.add_argument('--profile', type=str, default="Train", help='config profile')
        parser.add_argument('--devices', type=int, nargs='+', default=[0], help='devices')
        parser.add_argument('--resume', type=str, default=None, help='resume training and provide the config path')
        parser.add_argument('--nofworkers', type=int, default=28)
        parser.add_argument('--local', action='store_true', help='training on local machine with small memory size and '
                                                                 'small gpu power')
        parsed, unknown = parser.parse_known_args()
        for arg in unknown:
            if arg.startswith(("-", "--")):
                parser.add_argument(arg, type=eval)
        return parser.parse_args()

    def prepare_and_get_configpath(self, args):
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
        return config_path

    def get_outpath_and_print_infos(self, config_path: str, args):
        outpath = os.path.dirname(config_path)
        #checkpoint_path = os.path.join(outpath, "checkpoint.tar")
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
        return outpath, log

    def load_profiles(self, config_path):
        starttime = time.time()
        import_config_util = ImportConfigUtil()
        experconfig = import_config_util.import_module(config_path)
        datasetconfig = experconfig.DatasetConfig()
        train_profile = datasetconfig.get_train_profile()
        progressprof = datasetconfig.get_progress()
        print("Config loaded ({:.2f} s)".format(time.time() - starttime))
        return train_profile, progressprof

    def build_datasets(self, trainprofile, progressprof, args):
        env_utils = EnvUtils()
        # build dataset & testing dataset
        starttime = time.time()
        testdataset = progressprof.get_dataset()

        batch_size_training = trainprofile.get_batchsize()
        batch_size_test = progressprof.get_batchsize()
        nof_workers = args.nofworkers
        if env_utils.is_local_env():
            batch_size_training = 3
            batch_size_test = 3
            nof_workers = 1

        if len(testdataset) <= 0:
            raise Exception("problem appeared get_dataset")
        test_dataloader = torch.utils.data.DataLoader(testdataset,
                                                      batch_size=batch_size_test,
                                                      shuffle=True,
                                                      drop_last=True,
                                                      num_workers=nof_workers)

        dataset = trainprofile.get_dataset()
        print("Train-Batchsize: {}".format(batch_size_training))
        print("Nof-Workers: {}".format(nof_workers))
        train_dataloader = torch.utils.data.DataLoader(dataset,
                                                       batch_size=batch_size_training,
                                                       shuffle=True,
                                                       drop_last=True,
                                                       num_workers=nof_workers)
        for item in dataset:
            print("Image Resolution Loss Img: {}".format(item['image'].shape))
            print("Image Resolution Encoder Input Img: {}".format(item['fixedcamimage'].shape))
            break
        print("Dataset instantiated ({:.2f} s)".format(time.time() - starttime))
        return train_dataloader, test_dataloader, dataset

    def get_writer_autencoder_optimizer_lossweights(self, trainprofile, progressprof, dataset, args, outpath):
        # data writer
        starttime = time.time()
        writer = progressprof.get_writer()
        print("Writer instantiated ({:.2f} s)".format(time.time() - starttime))

        # build autoencoder
        starttime = time.time()
        ae = trainprofile.get_autoencoder(dataset)
        torch.cuda.set_device(args.devices[0])
        ae = torch.nn.DataParallel(ae, device_ids=args.devices).to("cuda").train()
        if args.resume is not None:
            ae.module.load_state_dict(torch.load("{}/aeparams.pt".format(outpath)), strict=False)

        print("Autoencoder instantiated ({:.2f} s)".format(time.time() - starttime))

        # build optimizer
        starttime = time.time()
        aeoptim = trainprofile.get_optimizer(ae.module)
        lossweights = trainprofile.get_loss_weights()
        print("Optimizer instantiated ({:.2f} s)".format(time.time() - starttime))
        return writer, ae, aeoptim, lossweights

    def save_wandb_info(
            self,
            iternum,
            train_loss,
            train_output,
            test_loss,
            test_output,
            validation_img,
            wandb
    ):
        dict_wandb = {
            "train_loss": float(train_loss.item()),
            "test_loss": float(test_loss.item()),
            "step": iternum
        }
        dict_wandb = self.__append_wandb_dict_from_model_output(train_output, "train", dict_wandb)
        dict_wandb = self.__append_wandb_dict_from_model_output(test_output, "test", dict_wandb)
        if validation_img is not None:
            image = wandb.Image(validation_img, caption="Validation after {} Iterations".format(iternum))
            dict_wandb["validation_pictures"] = image
        wandb.log(dict_wandb)

    def __append_wandb_dict_from_model_output(self, output: dict, train_test_label: str, dict_wandb: dict):
        for k, v in output["losses"].items():
            label = "{}_{}".format(train_test_label, k)
            if isinstance(v, tuple):
                dict_wandb[label] = float(torch.sum(v[0]) / torch.sum(v[1]))
            else:
                dict_wandb[label] = float(torch.mean(v))
        return dict_wandb

    def print_iteration_infos(self, iternum, loss, output, starttime):
        print("Iteration {}: loss = {:.5f}, ".format(iternum, float(loss.item())) +
              ", ".join(["{} = {:.5f}".format(k, float(
                  torch.sum(v[0]) / torch.sum(v[1]) if isinstance(v, tuple) else torch.mean(v)))
                         for k, v in output["losses"].items()]), end="")
        if iternum % 5 == 0:
            endtime = time.time()
            ips = 10. / (endtime - starttime)
            print(", iter/sec = {:.2f}".format(ips))
            starttime = time.time()
        print("")
        return starttime

    def calculate_final_loss_from_output(self, output: dict, lossweights: dict):
        # compute final loss
        return sum([
            lossweights[k] * (torch.sum(v[0]) / torch.sum(v[1]) if isinstance(v, tuple) else torch.mean(v))
            for k, v in output["losses"].items()])

    def get_testbatch_testoutput(self, iternum, progressprof, test_dataloader, ae, lossweights):
        testoutput = None
        test_batch = None
        for test_batch in test_dataloader:
            testoutput = ae(
                iternum,
                lossweights.keys(),
                **{k: x.to("cuda") for k, x in test_batch.items()},
                **progressprof.get_ae_args())
            break

        if testoutput is None or test_batch is None:
            raise Exception("Should always have a test-batch and testoutput")
        return test_batch, testoutput

    def save_model_and_validation_pictures(
            self,
            iternum,
            outpath,
            ae,
            test_batch,
            testoutput,
            trainprofile,
            data,
            writer
    ) -> (np.array, torch.Tensor):
        np_img = None
        # save intermediate results
        if iternum % 1000 == 0 or iternum in [0, 1, 2, 3, 4, 5]:
            torch.save(ae.module.state_dict(), "{}/aeparams.pt".format(outpath))
            b = data["campos"].size(0)
            np_img = writer.batch(
                iternum,
                iternum * trainprofile.get_batchsize() + torch.arange(b),
                outpath,
                **test_batch,
                **testoutput)
        return np_img

    def debug_memory_info(self, label:str):
        device = "cuda:0"
        t = torch.cuda.get_device_properties(device).total_memory
        r = torch.cuda.memory_reserved(device)
        a = torch.cuda.memory_allocated(device)
        f = r - a  # free inside reserved
        print("{} - free memory: {}".format(label, f))
