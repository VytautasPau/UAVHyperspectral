import random
import torch
import numpy as np
import argparse
import datasets
from models import AutoEncoder, NeuralNet
from estimator import NeuralNetEstimator
import os
from sklearn.model_selection import ParameterGrid

seed = int(np.random.randint(0, 10000, 1)[0])
print(seed)
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


class Dataset:
    def __init__(self, dataset, device="cpu", split=None, model="unet", root_folder=""):
        self.dataset = dataset
        self.device = device
        self.root_folder = root_folder
        if dataset == 'samson':
            self.LR, self.EPOCH = 6e-3, 200
            self.patch, self.dim = 5, 200
            self.beta, self.gamma, self.delta = 1, 1, 1
            self.weight_decay_param = 4e-5
            self.order_abd, self.order_endmem = (0, 1, 2), (0, 1, 2)
            if model == "transformer":
                self.data = datasets.Data(dataset, self.device, split=split, patch_size=self.patch, root_folder=self.root_folder)
            else:
                self.data = datasets.Data(dataset, self.device, split=split, root_folder=self.root_folder)
            self.loader = self.data.get_loader()
            self.init_weight = self.data.get("init_weight").unsqueeze(2).unsqueeze(3).float()
            self.part_shape = 16
        elif dataset == 'apex':
            self.LR, self.EPOCH = 9e-3, 200
            self.patch, self.dim = 5, 200
            self.beta, self.gamma, self.delta = 1, 0.01, 1
            self.weight_decay_param = 4e-5
            self.order_abd, self.order_endmem = (3, 1, 2, 0), (3, 1, 2, 0)
            if model == "transformer":
                self.data = datasets.Data(dataset, self.device, split=split, patch_size=self.patch, root_folder=self.root_folder)
            else:
                self.data = datasets.Data(dataset, self.device, split=split, root_folder=self.root_folder)
            self.loader = self.data.get_loader()
            self.init_weight = self.data.get("init_weight").unsqueeze(2).unsqueeze(3).float()
            self.part_shape = 16
        elif dataset == 'dc':
            self.LR, self.EPOCH = 0.0003, 3000
            self.patch, self.dim = 2, 400
            self.beta, self.gamma, self.delta = 1, 1, 1
            self.weight_decay_param = 3e-5
            self.order_abd, self.order_endmem = (0, 2, 1, 5, 4, 3), (0, 2, 1, 5, 4, 3)
            if model == "transformer":
                self.data = datasets.Data(dataset, self.device, split=split, patch_size=self.patch, root_folder=self.root_folder)
            else:
                self.data = datasets.Data(dataset, self.device, split=split, root_folder=self.root_folder)
            self.loader = self.data.get_loader()
            self.init_weight = self.data.get("init_weight").unsqueeze(2).unsqueeze(3).float()
            self.part_shape = 16
        elif dataset == "new":
            self.LR, self.EPOCH = 0.001, 3000
            self.patch, self.dim = 8, 128
            self.beta, self.gamma, self.delta = 1, 0.1, 0.05
            self.weight_decay_param = 3e-5
            if model == "transformer":
                self.data = datasets.Data(dataset, self.device, original=False, split=split, patch_size=self.patch, root_folder=self.root_folder)
                self.beta, self.gamma, self.delta = 1, 0.0001, 0.05
                self.patch, self.dim = 4, 400
            else:
                self.data = datasets.Data(dataset, self.device, original=False, split=split, root_folder=self.root_folder)
            self.loader = self.data.get_loader()
            self.order_abd, self.order_endmem = tuple(list(range(self.data.C))), tuple(list(range(self.data.C)))
            self.init_weight = self.data.get("init_weight").unsqueeze(2).unsqueeze(3).float()
            self.part_shape = 32 
        else:
            raise ValueError("Unknown dataset")
        self.row = self.data.row
        self.col = self.data.col
        self.L = self.data.L
        self.C = self.data.C

    def __str__(self):
        return f"{self.dataset} - cube shape: {self.data.Y.shape} - abd shape {self.data.A.shape}"


class Experiment:
    def __init__(self, model: str, device: str, dataset: str, summary: bool, out_dir: str,
                 epochs: int, split: bool, root_folder: str, grid_serach=False, start_iter=0, end_iter=0):
        self.model_name = model
        self.dataset = dataset
        self.device = device
        self.summary = summary
        self.epochs = epochs
        self.out_dir = out_dir  # base experiment out dir
        self.base_out_dir = out_dir
        self.split = split
        self.root_folder = root_folder
        self.grid_search = grid_serach
        self.start_iter = start_iter
        self.end_iter = end_iter

        if self.grid_search:
            param_grid = {"classes": [6], "beta": [0.1], "gamma": [1, 0.1], "part_shape": [16, 20, 24, 28, 32, 40, 48, 54, 64], "LR": [0.001, 0.005]}
            self.param_combinations = list(ParameterGrid(param_grid))
            self.iterator = self.start_iter
            self._gather_dataset()
        else:
            self.parameters = None
            self._gather_dataset()
            self._gather_parameters()
            self._gen_out_dir()
            self._gather_model()

    def _gather_dataset(self, n_class=None):
        # possible lazy loading could be implemented if required
        self.dt = Dataset(self.dataset, self.device, split=self.split, model=self.model_name, root_folder=self.root_folder)
        if n_class is not None:
            self.dt.C = n_class

    def _gather_parameters(self, param_override=None):
        # Gather parameters for the dataset into a dict
        self.parameters = {"classes": self.dt.C, "wavelengths": self.dt.L, "col": self.dt.col,
                           "row": self.dt.row, "patch": self.dt.patch, "dim": self.dt.dim, "part_shape": self.dt.part_shape,
                           "beta": self.dt.beta, "gamma": self.dt.gamma, "delta": self.dt.delta, "LR": self.dt.LR}
        if param_override is not None:
            self.parameters.update(param_override)

    def _gen_out_dir(self, iteration=None):
        if iteration is not None:
            self.out_dir = os.path.join(self.base_out_dir, f"{self.model_name}_{self.dataset}_gs_{iteration + 1}")
        else:
            self.out_dir = os.path.join(self.base_out_dir, f"{self.model_name}_{self.dataset}")
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def _gather_model(self):
        if self.parameters is None:
            self._gather_parameters()
        self.model = NeuralNet(self.parameters["wavelengths"], self.parameters["classes"], self.parameters["col"], self.parameters["row"],
                               out_dir=self.out_dir, WL=self.dt.data.WL.squeeze(), beta=self.parameters["beta"], epochs=self.epochs,
                               gamma=self.parameters["gamma"], delta=self.parameters["delta"], LR=self.parameters["LR"], 
                               dim=self.parameters["dim"], patch=self.parameters["patch"], part_shape=self.parameters["part_shape"],
                               init_weight=self.dt.init_weight)
        print("Loading model")
        self.model.load_model(self.model_name)
        self._get_estimator()

    def _get_estimator(self):
        self.estimator = NeuralNetEstimator(self.model, self.device, self.summary)

    def run(self):
        if self.summary:
            return False
        if self.grid_search:
            end_iter = len(self.param_combinations)
            if self.end_iter > 0:
                end_iter = self.end_iter
            while True:
                if self.iterator >= end_iter:
                    break
                curr_params = self.param_combinations[self.iterator]
                self._gather_parameters(curr_params)
                self._gather_dataset(self.parameters["classes"])
                self._gen_out_dir(self.iterator)
                self.model = None
                self._gather_model()
                print(f"Starting grid serach iteration {self.iterator + 1} out of {len(self.param_combinations)}")
                print(self.parameters)
                self.model.reconst_out_epoch = 0
                # run (2 - 1) tests during training
                self.model.test_epoch = int(self.epochs // 2)
                self.estimator.fit(self.dt)
                self.model.full_test(self.dt, self.iterator + 1, self.parameters)
                # increase iteration and continue
                self.iterator += 1
        else:
            self.estimator.fit(self.dt)
            self.estimator.predict(self.dt)


def main(params):
    exp = Experiment(params["model"], params["gpus"][0], params["datasets"][0], params["summary"], params["out"], params["epochs"],
                     params["split"], params["root_folder"], params["search"], params["start_iter"], params["end_iter"])
    exp.run()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out', type=str, required=True, help='Relative path to output folder')
    parser.add_argument('-f', '--folder', dest="root_folder", type=str, required=True, help='Absolute path to root dataset folder')
    parser.add_argument('-d', '--datasets', nargs='+', help='List of datasets to run experiments on', default=["new2-split"])
    parser.add_argument('--model', type=str, default="unet", help='Select model to be used')
    parser.add_argument('-g', '--gpus', nargs='+', help='List of gpus to use', default=["cuda:0"])
    parser.add_argument('--jobs', type=int, default=1, help='Number of concurent jobs (ideally a multiple of # of gpus)')
    parser.add_argument('--summary', default=False, action='store_true')
    parser.add_argument('--search', default=False, action='store_true')
    parser.add_argument('--split', default=None, type=str)
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--start_iter', type=int, default=0, help='Grid search iteration to start form')
    parser.add_argument('--end_iter', type=int, default=0, help='Grid search iteration to end on (0 end on len)')

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    main(vars(opt))
