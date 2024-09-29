import unittest
from datasets import Data
import sys
from experiment import Dataset, Experiment


class DataTest(unittest.TestCase):
    device = 'cpu'
    root_folder = "/home/vytas/SynologyDrive/Doktorantura/Full-dataset"
    
    def test_data_init(self):  #init with nonexistatnt dataset
        data = Data(dataset="test", device=self.device, original=True)
        self.assertIsNone(data.C)
        self.assertIsNone(data.L)
        self.assertIsNone(data.col)
        self.assertIsNone(data.row)
        self.assertIsNone(data.WL)

    def test_original_dc(self):
        data = Data(dataset="dc", device=self.device, original=True)
        self.assertEqual(data.C, 6)
        self.assertEqual(data.L, 191)
        self.assertEqual(data.col, 290)
        self.assertEqual(data.row, 290)
        self.assertEqual(data.Y.shape[0], data.L)
        self.assertEqual(data.Y.shape[1], data.col)
        self.assertEqual(data.Y.shape[2], data.row)
        self.assertEqual(data.A.shape[0], data.C)

    def test_original_apex(self):
        data = Data(dataset="apex", device=self.device, original=True)
        self.assertEqual(data.C, 4)
        self.assertEqual(data.L, 285)
        self.assertEqual(data.col, 110)
        self.assertEqual(data.row, 110)
        self.assertEqual(data.Y.shape[0], data.L)
        self.assertEqual(data.Y.shape[1], data.col)
        self.assertEqual(data.Y.shape[2], data.row)
        self.assertEqual(data.A.shape[0], data.C)

    def test_original_samson(self):
        data = Data(dataset="samson", device=self.device, original=True)
        self.assertEqual(data.C, 3)
        self.assertEqual(data.L, 156)
        self.assertEqual(data.col, 95)
        self.assertEqual(data.row, 95)
        self.assertEqual(data.Y.shape[0], data.L)
        self.assertEqual(data.Y.shape[1], data.col)
        self.assertEqual(data.Y.shape[2], data.row)
        self.assertEqual(data.A.shape[0], data.C)

    def test_new_dataset(self):
        data = Data(dataset="new", device=self.device, original=False, root_folder=self.root_folder)
        self.assertEqual(data.C, 6)
        self.assertEqual(data.L, 224)
        self.assertEqual(data.col, 341)
        self.assertEqual(data.row, 938)
        self.assertEqual(data.WL.shape[0], data.L)
        self.assertEqual(data.Y.shape[0], data.L)
        self.assertEqual(data.Y.shape[1], data.col)
        self.assertEqual(data.Y.shape[2], data.row)
        self.assertEqual(data.A.shape[0], data.C)

    def test_new_dataset_col_split(self):
        data = Data(dataset="new", device=self.device, original=False, root_folder=self.root_folder, split="col")
        self.assertEqual(data.C, 6)
        self.assertEqual(data.L, 224)
        #self.assertEqual(data.col, 341)
        self.assertEqual(data.row, 938)
        self.assertEqual(data.WL.shape[0], data.L)
        self.assertEqual(data.Y.shape[0], data.L)
        self.assertEqual(data.Y.shape[1], data.col)
        self.assertEqual(data.Y.shape[2], data.row)
        self.assertEqual(data.A.shape[0], data.C)

    def test_new_dataset_row_split(self):
        data = Data(dataset="new", device=self.device, original=False, root_folder=self.root_folder, split="row")
        self.assertEqual(data.C, 6)
        self.assertEqual(data.L, 224)
        self.assertEqual(data.col, 341)
        #self.assertEqual(data.row, 1059)
        self.assertEqual(data.WL.shape[0], data.L)
        self.assertEqual(data.Y.shape[0], data.L)
        self.assertEqual(data.Y.shape[1], data.col)
        self.assertEqual(data.Y.shape[2], data.row)
        self.assertEqual(data.A.shape[0], data.C)

    def test_new_dataset_patch_4(self):
        data = Data(dataset="new", device=self.device, original=False, root_folder=self.root_folder, patch_size=4)
        self.assertEqual(data.C, 6)
        self.assertEqual(data.L, 224)
        self.assertEqual(data.col, 340)
        self.assertEqual(data.row, 936)
        self.assertEqual(data.WL.shape[0], data.L)
        self.assertEqual(data.Y.shape[0], data.L)
        self.assertEqual(data.Y.shape[1], data.col)
        self.assertEqual(data.Y.shape[2], data.row)
        self.assertEqual(data.A.shape[0], data.C)

    def test_new_dataset_patch_32(self):
        data = Data(dataset="new", device=self.device, original=False, root_folder=self.root_folder, patch_size=32)
        self.assertEqual(data.C, 6)
        self.assertEqual(data.L, 224)
        self.assertEqual(data.col, 320)
        self.assertEqual(data.row, 928)
        self.assertEqual(data.WL.shape[0], data.L)
        self.assertEqual(data.Y.shape[0], data.L)
        self.assertEqual(data.Y.shape[1], data.col)
        self.assertEqual(data.Y.shape[2], data.row)
        self.assertEqual(data.A.shape[0], data.C)

    def test_dataloader(self):
        data = Data(dataset="new", device=self.device, original=False, root_folder=self.root_folder, patch_size=32)
        dl = data.get_loader()
        dat, _ = next(iter(dl))
        self.assertEqual(len(dat.shape), 4) # batch dimmension added
        self.assertEqual(dat.shape[0], 1)
        self.assertEqual(dat.shape[1], data.L)
        self.assertEqual(dat.shape[2], data.col)
        self.assertEqual(dat.shape[3], data.row)


class DatasetTest(unittest.TestCase):
    device = 'cpu'
    root_folder = "/home/vytas/SynologyDrive/Doktorantura/Full-dataset"

    def test_new(self):
        dataset = Dataset(dataset="new", root_folder=self.root_folder)
        self.assertEqual(dataset.data.Y.shape[0], dataset.L)
        self.assertEqual(dataset.data.Y.shape[1], dataset.col)
        self.assertEqual(dataset.data.Y.shape[2], dataset.row)
        self.assertEqual(dataset.data.A.shape[0], dataset.C)
        dl = dataset.data.get_loader()
        dat, _ = next(iter(dl))
        self.assertEqual(len(dat.shape), 4) # batch dimmension added
        self.assertEqual(dat.shape[0], 1)
        self.assertEqual(dat.shape[1], dataset.L)
        self.assertEqual(dat.shape[2], dataset.col)
        self.assertEqual(dat.shape[3], dataset.row)


class ExperimentTest(unittest.TestCase):
    epochs = 5
    root_folder = "/home/vytas/SynologyDrive/Doktorantura/Full-dataset"

    def test_exp_unet_new(self):
        exp = Experiment(model="unet", device="cpu", dataset="new", summary=False, out_dir="test_out", epochs=self.epochs, split=None, root_folder=self.root_folder)
        exp.run()

    def test_exp_unet_dc(self):
        exp = Experiment(model="unet", device="cpu", dataset="dc", summary=False, out_dir="test_out", epochs=self.epochs, split=None, root_folder=self.root_folder)
        exp.run()

    def test_exp_unet_new_gpu(self):
        exp = Experiment(model="unet", device="cuda:0", dataset="new", summary=False, out_dir="test_out", epochs=self.epochs, split=None, root_folder=self.root_folder)
        exp.run()

    def test_exp_unet_dc_gpu(self):
        exp = Experiment(model="unet", device="cuda:0", dataset="dc", summary=False, out_dir="test_out", epochs=self.epochs, split=None, root_folder=self.root_folder)
        exp.run()

    def test_exp_transformer_new(self):
        exp = Experiment(model="transformer", device="cpu", dataset="new", summary=False, out_dir="test_out", epochs=self.epochs, split=None, root_folder=self.root_folder)
        exp.run()

    def test_exp_transformer_dc(self):
        exp = Experiment(model="transformer", device="cpu", dataset="dc", summary=False, out_dir="test_out", epochs=self.epochs, split=None, root_folder=self.root_folder)
        exp.run()
    # WIP test for more epochs and check rgb, abd and endm intermediate and final results file creation
    # delete test/out after tests arte complete

    
if __name__ == "__main__":
    if len(sys.argv) > 0:
        DataTest.device = sys.argv.pop()
    else:
        DataTest.device = 'cpu'
    unittest.main()
