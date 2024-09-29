import numpy as np
from experiment import Dataset
from utils import loss_simmilarity


def Main():
    fld ="/home/vytas/SynologyDrive/Doktorantura/Full-dataset"
    dataset = "new"
    dt = Dataset(dataset, root_folder=fld)

    dat = dt.data.get("end_mem")
    print(dat.shape)
    print(loss_simmilarity(dat.T))
    print(loss_simmilarity(dat, norm=2))
    print(loss_simmilarity(dat, norm=1))
    print(loss_simmilarity(dat, norm=100))


if __name__ == "__main__":
    Main()
