import numpy as np
from matplotlib import pyplot as plt


def plot_abundance(ground_truth, estimated, em, save_dir, epoch=0, loss=0, dataset="val", name_ext=""):
    print(ground_truth.shape, estimated.shape, em)
    plt.figure(figsize=(40, 5), dpi=300)
    for i in range(em):
         
        plt.subplot(2, em, i + 1)
        if i >= ground_truth.shape[2]:
            plt.imshow(np.zeros_like(ground_truth[:, :, 0]), cmap='jet', vmin=0, vmax=1)
        else:
            plt.imshow(ground_truth[:, :, i], cmap='jet', vmin=0, vmax=1)
        plt.axis('off')

    for i in range(em):
        plt.subplot(2, em, em + i + 1)
        plt.imshow(estimated[:, :, i], cmap='jet', vmin=0, vmax=1)
        plt.axis('off')
    plt.tight_layout()
    if epoch != 0:
        plt.savefig(save_dir + "abundance_" + str(epoch) + "_" + str(loss) + "_" + dataset + ".png")
    else:
        plt.savefig(save_dir + f"abundance{name_ext}.png")
    plt.close()


def plot_endmembers(target, pred, em, save_dir, epoch=0, loss=0, dataset="val", name_ext = "", errors=None):

    plt.figure(figsize=(40, 5), dpi=300)
    for i in range(em):
        plt.subplot(2, em // 2 if em % 2 == 0 else em, i + 1)
        plt.plot(pred[:, i], label="Extracted")
        if i >= target.shape[1]:
            plt.plot(np.zeros_like(target[:, 0]), label="GT")
        else:
            plt.plot(target[:, i], label="GT")
            if errors is not None:
                plt.fill_between(np.arange(target.shape[0]), target[:, i]-errors[:, i], target[:, i]+errors[:, i], alpha=0.5, color="green")
        plt.legend(loc="upper left")
        # plt.axis('off')
    plt.tight_layout()

    if epoch != 0:
        plt.savefig(save_dir + "end_members_" + str(epoch) + "_" + str(loss) + "_" + dataset + ".png")
    else:
        plt.savefig(save_dir + f"end_members{name_ext}.png")
    plt.close()
