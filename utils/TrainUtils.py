import argparse
import os
import shutil
from datetime import datetime
import time
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
        parser.add_argument('--nofworkers', type=int, default=16)
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
        batch_size_test = progressprof.get_batchsize
        if env_utils.is_local_env():
            batch_size_test = 3
        if len(testdataset) <= 0:
            raise Exception("problem appeared get_dataset")
        dataloader = torch.utils.data.DataLoader(testdataset, batch_size=batch_size_test, shuffle=False, drop_last=True,
                                                 num_workers=0)
        if len(dataloader) <= 0:
            raise Exception("problem appeared dataloader")
        for testbatch in dataloader:
            break

        dataset = trainprofile.get_dataset()
        nof_workers = args.nofworkers
        batch_size_training = trainprofile.get_batchsize()
        if env_utils.is_local_env():
            nof_workers = 1
            batch_size_training = 3
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_training, shuffle=True, drop_last=True,
                                                 num_workers=nof_workers)
        print("Dataset instantiated ({:.2f} s)".format(time.time() - starttime))
        return dataloader, testbatch, dataset

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
