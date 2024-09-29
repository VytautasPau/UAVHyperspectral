import torch
import torch.nn as nn
import numpy as np


def compute_rmse(x_true, x_pre):
    w, h, c = x_true.shape
    class_rmse = [0] * c
    for i in range(c):
        class_rmse[i] = np.sqrt(((x_true[:, :, i] - x_pre[:, :, i]) ** 2).sum() / (w * h))
    if c < x_pre.shape[-1]:
        mean_rmse = np.sqrt(((x_true - x_pre[:, :, :x_true.shape[-1]]) ** 2).sum() / (w * h * c))
    else:
        mean_rmse = np.sqrt(((x_true - x_pre) ** 2).sum() / (w * h * c))
    return class_rmse, mean_rmse


def compute_re(x_true, x_pred):
    img_w, img_h, img_c = x_true.shape
    return np.sqrt(((x_true - x_pred) ** 2).sum() / (img_w * img_h * img_c))


def compute_sad(inp, target):
    p = inp.shape[-1]
    sad_err = [0] * p
    for i in range(p):
        inp_norm = np.linalg.norm(inp[:, i], 2)
        tar_norm = np.linalg.norm(target[:, i], 2)
        summation = np.matmul(inp[:, i].T, target[:, i])
        sad_err[i] = np.arccos(summation / (inp_norm * tar_norm))
        sad_err[i] = np.nan_to_num(sad_err[i])
    mean_sad = np.mean(sad_err)
    return sad_err, mean_sad


def Nuclear_norm(inputs):
    _, band, h, w = inputs.shape
    inp = torch.reshape(inputs, (band, h * w))
    out = torch.norm(inp, p='nuc')
    return out


class SparseKLloss(nn.Module):
    def __init__(self):
        super(SparseKLloss, self).__init__()

    def __call__(self, inp, decay):
        inp = torch.sum(inp, 0, keepdim=True)
        loss = Nuclear_norm(inp)
        return decay * loss


class SumToOneLoss(nn.Module):
    def __init__(self, device):
        super(SumToOneLoss, self).__init__()
        self.register_buffer('one', torch.tensor(1, dtype=torch.float, device=device))
        self.loss = nn.L1Loss(reduction='sum')

    def get_target_tensor(self, inp):
        target_tensor = self.one
        return target_tensor.expand_as(inp)

    def __call__(self, inp, gamma_reg):
        inp = torch.sum(inp, 1)
        target_tensor = self.get_target_tensor(inp)
        loss = self.loss(inp, target_tensor)
        return gamma_reg * loss


class SAD(nn.Module):
    def __init__(self, num_bands):
        super(SAD, self).__init__()
        self.num_bands = num_bands

    def forward(self, inp, target):
        # print("sad")
        # print(inp.shape, target.shape)
        # print(inp.view(-1, 1, self.num_bands).shape, inp.view(-1, self.num_bands, 1).shape)
        try:
            input_norm = torch.sqrt(torch.bmm(inp.view(-1, 1, self.num_bands),
                                              inp.view(-1, self.num_bands, 1)))
            target_norm = torch.sqrt(torch.bmm(target.view(-1, 1, self.num_bands),
                                               target.view(-1, self.num_bands, 1)))
 
            summation = torch.bmm(inp.view(-1, 1, self.num_bands), target.view(-1, self.num_bands, 1))
            angle = torch.acos(summation / (input_norm * target_norm))
            angle = torch.nan_to_num(angle)

        except ValueError:
            return 0.0

        return angle


class NewSAD(nn.Module):
    #  based on https://github.com/Lightning-AI/torchmetrics/blob/v1.3.1/src/torchmetrics/functional/image/sam.py
    def __init__(self, num_bands):
        super(NewSAD, self).__init__()
        self.num_bands = num_bands

    def forward(self, inp, target):
        # input shapes: [batch, pixels, bands]
        try:
            # calculate norms on spectral dim
            n1 = inp.norm(dim=2) 
            n2 = target.norm(dim=2) 
            summation = (inp * target).sum(dim=2)
 
            angle = torch.clamp(summation / (n1 * n2), -1, 1).acos()
            angle = torch.nan_to_num(angle)

        except ValueError:
            return 0.0

        return angle


class SID(nn.Module):
    def __init__(self, epsilon: float = 1e5):
        super(SID, self).__init__()
        self.eps = epsilon

    def forward(self, inp, target):
        normalize_inp = (inp / torch.sum(inp, dim=0)) + self.eps
        normalize_tar = (target / torch.sum(target, dim=0)) + self.eps
        sid = torch.sum(normalize_inp * torch.log(normalize_inp / normalize_tar) +
                        normalize_tar * torch.log(normalize_tar / normalize_inp))

        return sid


def loss_simmilarity(x, y=None, dim=0, norm="cos"):
    # minimize average magnitude of cosine similarity
    # performe pairwise endmember simmilarity calculation
    # input shape: [num of classes, bands]
    x = x.squeeze()
    nc = x.shape[0]
    x = x.T
    x_row = x[:, None, :].expand(-1, nc, -1)
    x_col = x[:, :, None].expand(-1, -1, nc)
    if norm == "cos":
        sim = nn.functional.cosine_similarity(x_col, x_row, dim=dim)
        return sim
    else:
        assert isinstance(norm, int)
        sim = nn.functional.pdist(x, p=norm)
        return sim

    #TODO: dissimilarity metric, Spearman coefficient, density, psi^2. Patestuot ant pradiniu klasiu.
    #I plota kiek pataiko kreives.
    # vizualizuoti klases i plokstumas, (PCA, MDS, T-SNE).
    # daugiau klasiu pabandyti. 
    # panaudoti klasiu sudarymo metrikas, mokymo metu.
    # kelis klasiu samples duot.
    

