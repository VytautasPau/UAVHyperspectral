import numpy as np
from experiment import main as exp_main


def main():

    # create base experiments for all datasets and models
    params = {"out": "demo6", "gpus": ["cuda:0"], "jobs": 1, "summary": False, "search": False, "split": None, "epochs": 3001, "start_iter": 0, "end_iter": 0}
    ### transformer model
    data_folder = "/home/vytas/SynologyDrive/Doktorantura/HU-codes/Main/data"
    # samson
    """
    params.update({"datasets": ["samson"], "model": "transformer", "root_folder": data_folder})
    exp_main(params)

    # apex
    params.update({"datasets": ["apex"], "model": "transformer", "root_folder": data_folder})
    exp_main(params)
    # dc
    params.update({"datasets": ["dc"], "model": "transformer", "root_folder": data_folder})
    exp_main(params)
    # update data folder to new dataset
    data_folder = "/home/vytas/SynologyDrive/Doktorantura/full-dataset-2"
    # New cubes
    params.update({"datasets": ["new"], "model": "transformer", "root_folder": data_folder})
    exp_main(params)
    ### unet model

    data_folder = "/home/vytas/SynologyDrive/Doktorantura/HU-codes/Main/data"
    # samson
    params.update({"datasets": ["samson"], "model": "unet", "root_folder": data_folder})
    exp_main(params)

    # apex
    params.update({"datasets": ["apex"], "model": "unet", "root_folder": data_folder})
    exp_main(params)

    # dc
    params.update({"datasets": ["dc"], "model": "unet", "root_folder": data_folder})
    exp_main(params)

    """
    # update data folder to new dataset
    data_folder = "/home/vytas/SynologyDrive/Doktorantura/full-dataset-2"
    # New cubes
    params.update({"datasets": ["new"], "model": "unet", "root_folder": data_folder})
    exp_main(params)


if __name__ == "__main__":
    main()

