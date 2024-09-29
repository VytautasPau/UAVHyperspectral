import torch.utils.data
import os
import scipy.io as sio
import torchvision.transforms as transforms
import numpy as np


class TrainData(torch.utils.data.Dataset):
    def __init__(self, img, target, transform=None, target_transform=None):
        self.img = img.float()
        self.target = target.float()

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.img, self.target


class Data:
    def __init__(self, dataset, device='cpu', original=True, split=None, order=[0, 1, 2], patch_size=1, root_folder=None):
        # split -> None, col, row
        # Init HSI data parameters: Num of classes, Num of bands, cols and rows
        self.C, self.L, self.col, self.row, self.WL = [None] * 5
        # init params
        # path to root dataset folder, not including original datasets used in the paper (https://github.com/preetam22n/DeepTrans-HSU/tree/main),
        # those are stored in the ./data folder (move if required)
        self.root = root_folder
        self.dataset = dataset
        self.original = original
        self.device = device
        self.split = split
        self.order = order
        self.patch_size = patch_size
        self.ValA = None
        
        self._data_load()
        if self.C is not None:
            if self.split is not None:
                self._splitter()
            else:
                self._val()
            self._patcher()
            self._to_torch()

    def _data_load(self):
        if self.original:
            wl_path = "./Results/Wavelengths.mat"
            wl_data = sio.loadmat(wl_path)
            data_path = "./data/" + self.dataset + "_dataset.mat"
            if self.dataset == 'samson':
                self.C, self.L, self.col, self.row = 3, 156, 95, 95
                self.WL = wl_data["lambda_Samson"]
            elif self.dataset == 'apex':
                self.C, self.L, self.col, self.row = 4, 285, 110, 110
                self.WL = wl_data["lambda_Apex"]
            elif self.dataset == 'dc':
                self.C, self.L, self.col, self.row = 6, 191, 290, 290
                self.WL = wl_data["lambda_DC"]

            # if provided dataset doenst exist, skip init
            try:
                data = sio.loadmat(data_path)
            except FileNotFoundError:
                print(f"File {data_path} does not exist")
                return None

            Y = data['Y']
            self.Y = Y.reshape(self.L, self.col, self.row)
            A = data['A']
            self.A = A.reshape(self.C, self.col, self.row)
            self.M = data['M']  #endmembers 
            self.M1 = data['M1']  #init weights
            del data
        else:
            # extend non original dataset with you own if adding new dataset. aka, add new dataset loading here
            if self.dataset == "new":
                # self.Y = np.load(os.path.join(root_folder, "Naujas dataset/Classification/cube_2/new_cube_3_0.npy"))
                # self.Y = np.load(os.path.join(root_folder, "Naujas dataset/Classification/cube_1/new_cube_3_0.npy"))
                # train, test, val - default order
                order = [2, 1, 0]
                cubes = ["cube_1/new_cube_3_0.npy", "cube_2/new_cube_3_0.npy", "cube_3/new_cube_3_0.npy"]
                self.Y = np.load(os.path.join(self.root, cubes[order[0]]))
                self.TestY = np.load(os.path.join(self.root, cubes[order[1]]))
                self.ValY = np.load(os.path.join(self.root, cubes[order[2]]))
                # Y.shape = (col, row, WL (P))

                # self.A = np.load(os.path.join(root_folder, "Naujas dataset/Classification/cube_2/abundances_3_0.npy"))
                # self.A = np.load(os.path.join(root_folder, "Naujas dataset/Classification/cube_1/abundances_3_0.npy"))
                abds = ["cube_1/abundances_3_0.npy", "cube_2/abundances_3_0.npy","cube_3/abundances_3_0.npy"] 
                self.A = np.load(os.path.join(self.root, abds[order[0]]))
                self.TestA = np.load(os.path.join(self.root, abds[order[1]]))
                self.ValA = np.load(os.path.join(self.root, abds[order[2]]))
                # A.shape = (col, row, ENDM (C))

                self.Y = np.transpose(self.Y, (2, 0, 1))  #  WL, col, row 
                self.TestY = np.transpose(self.TestY, (2, 0, 1))  #  WL, col, row 
                self.ValY = np.transpose(self.ValY, (2, 0, 1))  #  WL, col, row 

                self.A = np.transpose(self.A, (2, 0, 1))
                self.TestA = np.transpose(self.TestA, (2, 0, 1))
                self.ValA = np.transpose(self.ValA, (2, 0, 1))

                self.col = min(self.Y.shape[1], self.ValY.shape[1], self.TestY.shape[1])
                self.row = min(self.Y.shape[2], self.ValY.shape[2], self.TestY.shape[2])
                self.L = self.Y.shape[0]
                self.C = self.A.shape[0]
                self.A = self.A[:, :self.col, :self.row]
                self.Y = self.Y[:, :self.col, :self.row]
                self.ValA = self.ValA[:, :self.col, :self.row]
                self.ValY = self.ValY[:, :self.col, :self.row]
                self.TestA = self.TestA[:, :self.col, :self.row]
                self.TestY = self.TestY[:, :self.col, :self.row]
                print(self.A.shape, self.ValA.shape, self.TestA.shape)

                # self.M = np.load(os.path.join(root_folder, "Naujas dataset/Classification/cube_2/W_hyper_2.npy"))
                # self.M = np.load(os.path.join(root_folder, "Naujas dataset/Classification/W_hyper.npy"))
                self.M = np.load(os.path.join(self.root, "cube_1/W_hyper.npy"))
                self.M1 = np.zeros_like(self.M)
                self.ME = np.load(os.path.join(self.root, "cube_1/W_hyper_errors.npy"))
                self.WL = np.array([397.670000,  400.280000,  402.890000,  405.510000,  408.140000,  410.760000,
                      413.380000,  415.990000,  418.610000,  421.240000,  423.860000,  426.490000,
                      429.110000,  431.740000,  434.370000,  437.000000,  439.630000,  442.260000,
                      444.890000,  447.520000,  450.160000,  452.790000,  455.430000,  458.060000,
                      460.700000,  463.340000,  465.980000,  468.620000,  471.260000,  473.900000,
                      476.540000,  479.180000,  481.830000,  484.480000,  487.120000,  489.770000,
                      492.410000,  495.070000,  497.710000,  500.370000,  503.010000,  505.670000,
                      508.320000,  510.980000,  513.630000,  516.290000,  518.950000,  521.610000,
                      524.260000,  526.930000,  529.580000,  532.240000,  534.920000,  537.570000,
                      540.240000,  542.900000,  545.570000,  548.240000,  550.900000,  553.570000,
                      556.240000,  558.920000,  561.590000,  564.260000,  566.940000,  569.610000,
                      572.290000,  574.960000,  577.640000,  580.320000,  583.000000,  585.680000,
                      588.360000,  591.040000,  593.730000,  596.410000,  599.100000,  601.780000,
                      604.470000,  607.150000,  609.850000,  612.540000,  615.230000,  617.920000,
                      620.610000,  623.300000,  625.990000,  628.700000,  631.390000,  634.080000,
                      636.790000,  639.480000,  642.180000,  644.890000,  647.580000,  650.290000,
                      652.990000,  655.700000,  658.400000,  661.110000,  663.820000,  666.520000,
                      669.230000,  671.940000,  674.650000,  677.360000,  680.080000,  682.790000,
                      685.500000,  688.220000,  690.930000,  693.650000,  696.370000,  699.090000,
                      701.810000,  704.530000,  707.250000,  709.970000,  712.700000,  715.420000,
                      718.150000,  720.880000,  723.600000,  726.330000,  729.060000,  731.790000,
                      734.520000,  737.250000,  739.980000,  742.720000,  745.460000,  748.180000,
                      750.920000,  753.660000,  756.400000,  759.150000,  761.890000,  764.630000,
                      767.360000,  770.100000,  772.850000,  775.600000,  778.350000,  781.090000,
                      783.840000,  786.590000,  789.340000,  792.090000,  794.840000,  797.590000,
                      800.340000,  803.100000,  805.850000,  808.610000,  811.360000,  814.120000,
                      816.880000,  819.640000,  822.400000,  825.160000,  827.920000,  830.690000,
                      833.450000,  836.220000,  838.980000,  841.750000,  844.520000,  847.290000,
                      850.050000,  852.830000,  855.600000,  858.370000,  861.150000,  863.920000,
                      866.700000,  869.470000,  872.240000,  875.020000,  877.800000,  880.590000,
                      883.360000,  886.150000,  888.930000,  891.710000,  894.490000,  897.290000,
                      900.070000,  902.850000,  905.650000,  908.430000,  911.230000,  914.020000,
                      916.800000,  919.600000,  922.390000,  925.190000,  927.980000,  930.780000,
                      933.580000,  936.380000,  939.180000,  941.980000,  944.780000,  947.580000,
                      950.380000,  953.180000,  955.990000,  958.800000,  961.600000,  964.410000,
                      967.220000,  970.030000,  972.840000,  975.650000,  978.460000,  981.270000,
                      984.080000,  986.900000,  989.720000,  992.540000,  995.360000,  998.170000,
                     1000.990000, 1003.820000])

    def _splitter(self):
        if self.split != "col" and self.split != "row":
            print(f"Error Data.split  value {self.split} is not supported. Supported values: None, col, row")
        if self.split == "col":
            splits = int(self.col // 3) 
            new_col = splits * 3
            if new_col != self.col:
                self.Y = self.Y[:, :new_col, :]
                self.A = self.A[:, :new_col, :]
            self.col = splits
            for i, ord in enumerate(self.order):
                if i == 0:
                    self.Y = self.Y[:, ord*splits:(ord+1)*splits, :]
                    self.A = self.A[:, ord*splits:(ord+1)*splits, :]
                if i == 1:
                    self.ValY = self.Y[:, ord*splits:(ord+1)*splits, :]
                    self.ValA = self.A[:, ord*splits:(ord+1)*splits, :]
                if i == 2:
                    self.TestY = self.Y[:, ord*splits:(ord+1)*splits, :]
                    self.TestA = self.A[:, ord*splits:(ord+1)*splits, :]
        if self.split == "row":
            splits = int(self.row // 3) 
            new_row = splits * 3
            if new_row != self.row:
                self.Y = self.Y[:, :, :new_row]
                self.A = self.A[:, :, :new_row]
            self.row = splits
            for i, ord in enumerate(self.order):
                if i == 0:
                    self.Y = self.Y[:, :, ord*splits:(ord+1)*splits]
                    self.A = self.A[:, :, ord*splits:(ord+1)*splits]
                if i == 1:
                    self.ValY = self.Y[:, :, ord*splits:(ord+1)*splits]
                    self.ValA = self.A[:, :, ord*splits:(ord+1)*splits]
                if i == 2:
                    self.TestY = self.Y[:, :, ord*splits:(ord+1)*splits]
                    self.TestA = self.A[:, :, ord*splits:(ord+1)*splits]

    def _patcher(self):
        if self.patch_size != 1:
            patch_col = self.col % self.patch_size
            patch_row = self.row % self.patch_size
            if patch_row != 0:
                new_row = self.row - patch_row
            else:
                new_row = self.row
            if patch_col != 0:
                new_col = self.col - patch_col
            else:
                new_col = self.col
            self.A = self.A[:, :new_col, :new_row]
            self.Y = self.Y[:, :new_col, :new_row]
            self.ValA = self.ValA[:, :new_col, :new_row]
            self.ValY = self.ValY[:, :new_col, :new_row]
            self.TestA = self.TestA[:, :new_col, :new_row]
            self.TestY = self.TestY[:, :new_col, :new_row]
            self.col = new_col
            self.row = new_row

    def _val(self):
        if self.ValA is None:
            self.ValA = self.A
            self.TestA = self.A
            self.ValY = self.Y
            self.TestY = self.Y

    def _to_torch(self):
        self.Y = torch.from_numpy(self.Y).to(self.device) 
        self.A = torch.from_numpy(self.A).to(self.device) 
        self.ValA = torch.from_numpy(self.ValA).to(self.device) 
        self.TestA = torch.from_numpy(self.TestA).to(self.device) 
        self.ValY = torch.from_numpy(self.ValY).to(self.device) 
        self.TestY = torch.from_numpy(self.TestY).to(self.device) 
        self.M1 = torch.from_numpy(self.M1)    
        self.M = torch.from_numpy(self.M)
        if hasattr(self, "ME"):
            self.ME = torch.from_numpy(self.ME)

    def get(self, typ):
        if typ == "hs_img":
            return self.Y.float()
        elif typ == "abd_map":
            return self.A.float()
        elif typ == "abd_map_val":
            return self.ValA.float()
        elif typ == "abd_map_test":
            return self.TestA.float()
        elif typ == "end_mem":
            return self.M
        elif typ == "init_weight":
            return self.M1
        elif typ == "val_img":
            return self.ValY.float()
        elif typ == "test_img":
            return self.TestY.float()
        elif typ == "endm_var":
            return self.ME

        
    def get_loader(self, batch_size=1):
        train_dataset = TrainData(img=self.Y, target=self.A)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False)
        return train_loader

