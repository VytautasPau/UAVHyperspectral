import numpy as np
import torch
import torch.nn as nn
import plots
import transformer
#import unetmodel
import unetmodel_new as unetmodel
from torchinfo import summary as torchsummary
import os
from rgb import RGB
import matplotlib.image
import time
import scipy.io as sio
import utils


class NonZeroClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(1e-6, 1)


class AutoEncoder(nn.Module):
    def __init__(self, P, L, size, patch, dim, row=None):
        super(AutoEncoder, self).__init__()
        self.P, self.L, self.size, self.dim = P, L, size, dim
        if row is None:
            self.row = size
        else:
            self.row = row
        self.encoder = nn.Sequential(
            nn.Conv2d(L, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.LeakyReLU(),
            nn.Conv2d(64, (dim*P)//patch**2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d((dim*P)//patch**2, momentum=0.5),
        )

        self.vtrans = transformer.ViT(image_size=size, patch_size=patch, dim=(dim*P), depth=2,
                                      heads=8, mlp_dim=12, pool='cls', row=row)
        
        self.upscale = nn.Sequential(
            nn.Linear(dim, size * self.row),
        )
        
        self.smooth = nn.Sequential(
            nn.Conv2d(P, P, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Softmax(dim=1),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(P, L, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(),
        )

    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        # print(x.size())
        abu_est = self.encoder(x)
        # print(abu_est.size())
        cls_emb = self.vtrans(abu_est)
        # print(cls_emb.size())
        cls_emb = cls_emb.view(1, self.P, -1)
        # print(cls_emb.size())
        abu_est = self.upscale(cls_emb).view(1, self.P, self.size, self.row)
        # print(abu_est.size())
        abu_est = self.smooth(abu_est)
        # print(abu_est.size())
        re_result = self.decoder(abu_est)
        # print(re_result.size())
        return abu_est, re_result


class NeuralNet():
    def __init__(self, in_bands, out_class, col, row, out_dir, dim=5, patch=200, pool_kernel_size=2,
                 part_shape=32, reconst_out_epoch=25, epochs=100, LR=0.0001, weight_decay_param=4e-5,
                 WL=None, beta=1, gamma=1, delta=1, init_weight=None, test_epoch=100):
        self.L = in_bands
        self.P = out_class
        self.col = col
        self.row = row
        self.pool = pool_kernel_size
        self.dim = dim
        self.patch = patch
        self.part_shape = part_shape
        self.reconst_out_epoch = reconst_out_epoch  # if zero, skip rgb generation
        self.EPOCH = epochs
        self.LR = LR
        self.weight_decay_param = weight_decay_param
        self.out_dir = out_dir
        self.WL = WL
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.init_weight = init_weight
        self.test_epoch = test_epoch

    def load_model(self, model: str):
        self.model_name = model
        if self.model_name == "unet":
            self.model = unetmodel.UNet(self.L, self.P, self.part_shape, self.part_shape)
        if self.model_name == "transformer":
            self.model = AutoEncoder(P=self.P, L=self.L, size=self.col, patch=self.patch, dim=self.dim, row=self.row)

    def to(self, device):
        self.model = self.model.to(device)

    def summary(self):
        if self.model_name == "unet":
            torchsummary(self.model, (1, self.L, self.part_shape, self.part_shape))
            """
            # torchviz example
            tn = torch.rand((1, self.L, self.part_shape, self.part_shape))
            res = self.model(tn)
            grf = make_dot(res, params=dict(list(self.model.named_parameters())))
            resize_graph(grf, size_per_element=0.15, min_size=12)
            grf.render("rnn_torchviz", format="png")
            """
        if self.model_name == "transformer":
            torchsummary(self.model, (1, self.L, self.col, self.row), batch_dim=None)
        print(self.model.state_dict().keys())

    def run(self, X, y):
        if self.reconst_out_epoch > 0:
            # print initial data rgb images
            self.rgb(X.data.Y.detach().cpu().numpy(), "init", "train") 
            self.rgb(X.data.TestY.detach().cpu().numpy(), "init", "test") 
            self.rgb(X.data.ValY.detach().cpu().numpy(), "init", "val") 
        # X - experiments.Dataset class
        print(f"Training model: {self.model_name} for {self.EPOCH} epochs")
        if self.model_name == "unet":
            self._run_unet(X, y)
        if self.model_name == "transformer":
            self._run_vit(X, y)

    def test(self, X):
        print("Testing model: ", self.model_name)
        if self.model_name == "unet":
            self._test_unet(X, dataset="test")
        if self.model_name == "transformer":
            self._test_vit(X)

    def full_test(self, X, iteration, params):
        print("Running full test")
        self._full_test_unet(X, iteration, params)

    def _run_unet(self, X, y):
        self.model.apply(self.model.weights_init)

        model_dict = self.model.state_dict()
        # model_dict['decoder.0.weight'] = self.init_weight
        self.model.load_state_dict(model_dict)

        #loss_func = nn.MSELoss(reduction='mean')
        loss_func = nn.L1Loss()
        loss_func2 = nn.SmoothL1Loss()
        loss_func3 = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LR, weight_decay=self.weight_decay_param)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.LR*0.5, total_steps=self.EPOCH)
        apply_clamp_inst1 = NonZeroClipper()
        if self.reconst_out_epoch > 0:
            #self.rgb(np.transpose(X.data.Y.detach().cpu().numpy(), (1, 2, 0)), "init", 0, False)
            pass
        # train model
        time_start = time.time()
        self.model.train()
        self.epo_vs_los = []
        for epoch in range(self.EPOCH):
            for i, (x, _) in enumerate(X.loader):  # x.shape => batch, WL, col, row
                # permute col row and WL
                rollx = torch.tensor([0])
                rolly = torch.tensor([0])
                if self.reconst_out_epoch > 0:
                    if epoch % self.reconst_out_epoch != 0:
                        if torch.rand(1) > 0.5:
                            rollx = torch.randint(1, self.part_shape, (1,))
                            rolly = torch.randint(1, self.part_shape, (1,))
                            x = torch.roll(x, shifts=(rollx, rolly), dims=(2, 3))
                else:
                    if torch.rand(1) > 0.5:
                        rollx = torch.randint(1, self.part_shape, (1,))
                        rolly = torch.randint(1, self.part_shape, (1,))
                        x = torch.roll(x, shifts=(rollx, rolly), dims=(2, 3))
                tot_loss = 0
                tot_re = 0
                tot_sad = 0
                tot_abd = 0
                tot_sim = 0
                # print(x.size())  # 1, 224, col, row
                x_part = int(self.col // self.part_shape)
                y_part = int(self.row // self.part_shape)
                if self.col % self.part_shape != 0:
                    x_part += 1
                if self.row % self.part_shape != 0:
                    y_part += 1
                reconst = None
                if self.reconst_out_epoch > 0:
                    if epoch % self.reconst_out_epoch == 0 :
                        reconst = torch.clone(x)
                x_iter = np.arange(x_part)
                y_iter = np.arange(y_part)
                np.random.shuffle(x_iter)
                for l in x_iter:
                    min_x = self.part_shape*l if self.part_shape * (l+1) <= self.col else self.col - self.part_shape
                    max_x = self.part_shape*(l+1) if self.part_shape * (l+1) <= self.col else self.col
                    np.random.shuffle(y_iter)
                    for k in y_iter:
                        min_y = self.part_shape*k if self.part_shape * (k+1) <= self.row else self.row - self.part_shape
                        max_y = self.part_shape*(k+1) if self.part_shape * (k+1) <= self.row else self.row
                        
                        slice = x[:, :, min_x:max_x, min_y:max_y]
                        rnd = torch.rand(1)
                        if self.reconst_out_epoch > 0:
                            if rnd > 0.75 and epoch % self.reconst_out_epoch != 0:
                                if rnd > 0.75:
                                    slice = torch.rot90(slice, dims=[3, 2])
                                else:
                                    slice = torch.rot90(slice, dims=[2, 3])
                        else:
                            if rnd > 0.75:
                                if rnd > 0.75:
                                    slice = torch.rot90(slice, dims=[3, 2])
                                else:
                                    slice = torch.rot90(slice, dims=[2, 3])
                        # print(slice.size(), min_x, max_x, min_y, max_y, k)
                        abu_est, re_result, endm = self.model(slice)
                        if self.reconst_out_epoch > 0:
                            if epoch % self.reconst_out_epoch == 0:
                                reconst[:, :, min_x:max_x, min_y:max_y] = re_result
    
                        loss_re = self.beta * loss_func(re_result, slice)
                        loss_sad = loss_func2(re_result.view(1, self.L, -1).transpose(1, 2),
                                              slice.reshape(1, self.L, -1).transpose(1, 2))
                        loss_sad = self.gamma * torch.sum(loss_sad).float()
                        loss_abd = self.delta * loss_func3((re_result).view(1, self.L, self.part_shape, self.part_shape), slice)
                        loss_sim = utils.loss_simmilarity(endm).mean() 
                        # scale sim loss
                        loss_sim /= 10**(int(torch.log10(loss_sim)) - int(torch.log10(loss_re)) + 1)
     
                        total_loss = loss_re + loss_sad 
                        tot_loss += total_loss
                        tot_re += loss_re
                        tot_sad += loss_sad
                        tot_abd += loss_abd
                        tot_sim += loss_sim
                        # total_loss = loss_re

                        optimizer.zero_grad()
                        total_loss.backward()
                        # nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=1)
                        optimizer.step()

                # net.d3.apply(apply_clamp_inst1)
                if epoch % 5 == 0:
                    print('Epoch:', epoch, '| train loss: %.4f' % tot_loss.data,
                          '| re loss: %.4f' % tot_re.data,
                          '| sad loss: %.4f' % tot_sad.data,
                          '| abd loss: %.4f' % tot_abd.data,
                          '| sim loss: %.4f' % tot_sim.data,
                          '| LR: %.7f' % scheduler.get_last_lr()[0])
                if self.reconst_out_epoch > 0:
                    if epoch % self.reconst_out_epoch == 0:
                        self.rgb(reconst.detach().cpu().numpy(), epoch, float(tot_loss.data)) 
                if epoch % self.test_epoch == 0 and epoch > 0:
                    self._test_unet(X, epoch, float(tot_loss.data), "val")
                    self._test_unet(X, epoch, float(tot_loss.data), "train")
                    self.model.train()
                self.epo_vs_los.append(float(tot_loss.data))
            scheduler.step()

    def _run_vit(self, X, y):
        self.model.apply(self.model.weights_init)

        model_dict = self.model.state_dict()
        model_dict['decoder.0.weight'] = self.init_weight
        self.model.load_state_dict(model_dict)

        if self.reconst_out_epoch > 0:
            self.rgb(np.transpose(X.data.Y.detach().cpu().numpy(), (1, 2, 0)), "init", 0, False)
        # train model
        time_start = time.time()
        self.model.train()
        self.epo_vs_los = []

        loss_func = nn.MSELoss(reduction='mean')
        loss_func2 = utils.SAD(self.L)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LR, weight_decay=self.weight_decay_param)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
        apply_clamp_inst1 = NonZeroClipper()
        
        for epoch in range(self.EPOCH):
            for i, (x, _) in enumerate(X.loader):
                abu_est, re_result = self.model(x)

                loss_re = self.beta * loss_func(re_result, x)
                loss_sad = loss_func2(re_result.view(1, self.L, -1).transpose(1, 2),
                                      x.reshape(1, self.L, -1).transpose(1, 2))
                loss_sad = self.gamma * torch.sum(loss_sad).float()
                # print(loss_re, loss_sad)
                total_loss = loss_re + loss_sad

                optimizer.zero_grad()
                total_loss.backward()
                # nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=1)

                # printing grads
                # for param in net.parameters():
                #     if param.grad is not None:
                #         print(param.grad.size(), param.grad.mean())
                # print(re_result.mean())
                # print("\n \n ----- \n \n ")
                
                optimizer.step()

                self.model.decoder.apply(apply_clamp_inst1)
                
                if epoch % 10 == 0:
                    print('Epoch:', epoch, '| train loss: %.4f' % total_loss.data,
                          '| re loss: %.4f' % loss_re.data,
                          '| sad loss: %.4f' % loss_sad.data,
                          '| LR: %.7f' % scheduler.get_last_lr()[0])
                self.epo_vs_los.append(float(total_loss.data))
                if self.reconst_out_epoch > 0:
                    if epoch % self.reconst_out_epoch == 0:
                        self.rgb(re_result.detach().cpu().numpy(), epoch, float(total_loss.data)) 
                if epoch % self.test_epoch == 0 and epoch > 0:
                    self._test_vit(X, epoch, float(total_loss.data))
                    self.model.train()

            scheduler.step()
        time_end = time.time()
        
        print('Total computational cost:', time_end - time_start)


    def _test_unet(self, X, epoch=None, loss=0, dataset="val"):
        full_test = True
        col, row, P, L = self.col, self.row, self.P, self.L
        if dataset == "val":
            full_test = False
            print(f"Running partial test on epoch: {epoch}")
            x = X.data.get("val_img").unsqueeze(0)
        elif dataset == "test":
            x = X.data.get("test_img").unsqueeze(0)
        else:
            full_test = False
            print(f"Running partial test on epoch: {epoch} on train data" )
            x = X.data.get("hs_img").unsqueeze(0)


        # Testing ================

        self.model.eval()
        
        x_part = int(col // self.part_shape)
        y_part = int(row // self.part_shape)
        if col % self.part_shape != 0:
            x_part += 1
        if row % self.part_shape != 0:
            y_part += 1

        abu_est = torch.zeros_like(torch.empty(1, P, col, row))
        re_result = torch.zeros_like(x)
        endm = torch.zeros_like(torch.empty(L, P, x_part*y_part))
        i = 0
        for l in range(x_part):
            min_x = self.part_shape*l if self.part_shape * (l+1) <= col else col - self.part_shape
            max_x = self.part_shape*(l+1) if self.part_shape * (l+1) <= col else col
            for k in range(y_part):
                min_y = self.part_shape*k if self.part_shape * (k+1) <= row else row - self.part_shape
                max_y = self.part_shape*(k+1) if self.part_shape * (k+1) <= row else row
                
                ab, re, en = self.model(x[:, :, min_x:max_x, min_y:max_y])
                ab = ab.permute(0, 3, 1, 2)
                en = en.permute(1, 0)
                abu_est[:, :, min_x:max_x, min_y:max_y] = ab
                re_result[:, :, min_x:max_x, min_y:max_y] = re
                endm[:, :, i] = en
                i += 1
        
        abu_est = abu_est / (torch.sum(abu_est, dim=1))
        abu_est = abu_est.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        if dataset == "test":
            target = X.data.get("abd_map_test").permute(1, 2, 0).cpu().numpy()
        elif dataset == "val":
            target = X.data.get("abd_map_val").permute(1, 2, 0).cpu().numpy()
        else:
            target = X.data.get("abd_map").permute(1, 2, 0).cpu().numpy()

        true_endmem = X.data.get("end_mem").numpy()
        est_endmem = endm.detach().cpu().numpy()
        est_endmem = np.average(est_endmem, axis=-1)
        est_endmem = est_endmem.reshape((L, P))

        if X.dataset != "new":
            order = []
            for i in range(est_endmem.shape[1]):
                val = 10**5
                idx = 0
                for j in range(est_endmem.shape[1]):
                    if j in order:
                        continue
                    tmp = np.sqrt(np.mean((true_endmem[:, i] - est_endmem[:, j])**2))
                    if tmp < val:
                        val = tmp
                        idx = j
                order.append(idx)
            print(order)
        else:
            order = []
            vars = X.data.get('endm_var').numpy()
            for i in range(true_endmem.shape[1]):
                val = 0
                idx = 0
                for j in range(est_endmem.shape[1]):
                    if j in order:
                        continue
                    tmp = ((true_endmem[:, i] - est_endmem[:, j]) < vars[:, i]).sum()
                    if tmp > val:
                        val = tmp
                        idx = j
                order.append(idx)
            for j in range(est_endmem.shape[1]):
                if j not in order:
                    order.append(j)
            print("vars: ", order)

        abu_est = abu_est[:, :, order]
        est_endmem = est_endmem[:, order]

        if full_test:
            # sio.savemat(self.out_dir + f"/{X.dataset}_abd_map.mat", {"A_est": abu_est})
            # sio.savemat(self.out_dir + f"/{X.dataset}_endmem.mat", {"E_est": est_endmem})

            x = x.view(-1, col, row).permute(1, 2, 0).detach().cpu().numpy()
            re_result = re_result.view(-1, col, row).permute(1, 2, 0).detach().cpu().numpy()
            re = utils.compute_re(x, re_result)
            print("RE:", re)

            rmse_cls, mean_rmse = utils.compute_rmse(target, abu_est)
            print("Class-wise abundance RMSE value:")
            for i in range(len(rmse_cls)):
                print("Class", i + 1, ":", rmse_cls[i])
            print("Mean RMSE:", mean_rmse)

            sad_cls, mean_sad = utils.compute_rmse(true_endmem[:, np.newaxis, :], est_endmem[:, np.newaxis, :])
            print("Class-wise endmember RMSE value:")
            for i in range(len(sad_cls)):
                print("Class", i + 1, ":", sad_cls[i])
            print("Mean RMSE:", mean_sad)

            with open(self.out_dir + "/log1.csv", 'a') as file:
                file.write(f"LR: {self.LR}, ")
                file.write(f"WD: {self.weight_decay_param}, ")
                file.write(f"RE: {re:.4f}, ")
                file.write(f"SAD: {mean_sad:.4f}, ")
                file.write(f"RMSE: {mean_rmse:.4f}\n")

            plots.plot_abundance(target, abu_est, P, self.out_dir + "/")
            if X.dataset == "new":
                plots.plot_endmembers(true_endmem, est_endmem, P, self.out_dir + "/", errors=X.data.get('endm_var').numpy())
            else:
                plots.plot_endmembers(true_endmem, est_endmem, P, self.out_dir + "/")
        else:
            plots.plot_abundance(target, abu_est, P, self.out_dir + "/", epoch=epoch, loss=loss, dataset=dataset)
            if X.dataset == "new":
                plots.plot_endmembers(true_endmem, est_endmem, P, self.out_dir + "/", epoch=epoch, loss=loss, dataset=dataset, errors=X.data.get('endm_var').numpy())
            else:
                plots.plot_endmembers(true_endmem, est_endmem, P, self.out_dir + "/", epoch=epoch, loss=loss, dataset=dataset)


    def _test_vit(self, X, epoch=None, loss=0):
        full_test = True
        col, row, P, L = self.col, self.row, self.P, self.L
        if epoch is not None:
            full_test = False
            print(f"Running partial test on epoch: {epoch}")
            x = X.data.get("val_img").unsqueeze(0)
        else:
            x = X.data.get("test_img").unsqueeze(0)

        # Testing ================

        self.model.eval()

        abu_est, re_result = self.model(x)
        abu_est = abu_est / (torch.sum(abu_est, dim=1))
        abu_est = abu_est.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        if full_test:
            target = torch.reshape(X.data.get("abd_map_test"), (self.col, self.row, self.P)).cpu().numpy()
        else:
            target = torch.reshape(X.data.get("abd_map_val"), (self.col, self.row, self.P)).cpu().numpy()
        true_endmem = X.data.get("end_mem").numpy()
        est_endmem = self.model.state_dict()["decoder.0.weight"].cpu().numpy()
        est_endmem = est_endmem.reshape((self.L, self.P))

        if X.dataset != "new":
            order = []
            for i in range(est_endmem.shape[1]):
                val = 10**5
                idx = 0
                for j in range(est_endmem.shape[1]):
                    if j in order:
                        continue
                    tmp = np.sqrt(np.mean((true_endmem[:, i] - est_endmem[:, j])**2))
                    if tmp < val:
                        val = tmp
                        idx = j
                order.append(idx)
            print(order)
        else:
            order = []
            vars = X.data.get('endm_var').numpy()
            for i in range(true_endmem.shape[1]):
                val = 0
                idx = 0
                for j in range(est_endmem.shape[1]):
                    if j in order:
                        continue
                    tmp = ((true_endmem[:, i] - est_endmem[:, j]) < vars[:, i]).sum()
                    if tmp > val:
                        val = tmp
                        idx = j
                order.append(idx)
            for j in range(est_endmem.shape[1]):
                if j not in order:
                    order.append(j)
            order = list(set(order))
            print("vars: ", order)

        abu_est = abu_est[:, :, order]
        est_endmem = est_endmem[:, order]

        if full_test:
            sio.savemat(self.out_dir + f"/{X.dataset}_abd_map.mat", {"A_est": abu_est})
            sio.savemat(self.out_dir + f"/{X.dataset}_endmem.mat", {"E_est": est_endmem})

            x = x.view(-1, col, row).permute(1, 2, 0).detach().cpu().numpy()
            re_result = re_result.view(-1, col, row).permute(1, 2, 0).detach().cpu().numpy()
            re = utils.compute_re(x, re_result)
            print("RE:", re)

            rmse_cls, mean_rmse = utils.compute_rmse(target, abu_est)
            print("Class-wise RMSE value:")
            for i in range(P):
                print("Class", i + 1, ":", rmse_cls[i])
            print("Mean RMSE:", mean_rmse)

            sad_cls, mean_sad = utils.compute_sad(est_endmem, true_endmem)
            print("Class-wise SAD value:")
            for i in range(P):
                print("Class", i + 1, ":", sad_cls[i])
            print("Mean SAD:", mean_sad)

            with open(self.out_dir + "/log1.csv", 'a') as file:
                file.write(f"LR: {self.LR}, ")
                file.write(f"WD: {self.weight_decay_param}, ")
                file.write(f"RE: {re:.4f}, ")
                file.write(f"SAD: {mean_sad:.4f}, ")
                file.write(f"RMSE: {mean_rmse:.4f}\n")

            plots.plot_abundance(target, abu_est, P, self.out_dir + "/")
            plots.plot_endmembers(true_endmem, est_endmem, P, self.out_dir + "/")
        else:
            plots.plot_abundance(target, abu_est, P, self.out_dir + "/", epoch=epoch, loss=loss)
            plots.plot_endmembers(true_endmem, est_endmem, P, self.out_dir + "/", epoch=epoch, loss=loss)


    def _full_test_unet(self, X, iteration, params):
        col, row, P, L = self.col, self.row, self.P, self.L
        datasets = ["train", "test", "val"]
        tot_re = 0
        tot_rmse = 0
        tot_sad = 0
        for dataset in datasets:
            if dataset == "val":
                x = X.data.get("val_img").unsqueeze(0)
            elif dataset == "test":
                x = X.data.get("test_img").unsqueeze(0)
            else:
                x = X.data.get("hs_img").unsqueeze(0)

            # Testing ================

            self.model.eval()
            
            x_part = int(col // self.part_shape)
            y_part = int(row // self.part_shape)
            if col % self.part_shape != 0:
                x_part += 1
            if row % self.part_shape != 0:
                y_part += 1

            abu_est = torch.zeros_like(torch.empty(1, P, col, row))
            re_result = torch.zeros_like(x)
            endm = torch.zeros_like(torch.empty(L, P, x_part*y_part))
            i = 0
            for l in range(x_part):
                min_x = self.part_shape*l if self.part_shape * (l+1) <= col else col - self.part_shape
                max_x = self.part_shape*(l+1) if self.part_shape * (l+1) <= col else col
                for k in range(y_part):
                    min_y = self.part_shape*k if self.part_shape * (k+1) <= row else row - self.part_shape
                    max_y = self.part_shape*(k+1) if self.part_shape * (k+1) <= row else row
                    
                    ab, re, en = self.model(x[:, :, min_x:max_x, min_y:max_y])
                    ab = ab.permute(0, 3, 1, 2)
                    en = en.permute(1, 0)
                    abu_est[:, :, min_x:max_x, min_y:max_y] = ab
                    re_result[:, :, min_x:max_x, min_y:max_y] = re
                    endm[:, :, i] = en
                    i += 1
            
            abu_est = abu_est / (torch.sum(abu_est, dim=1))
            abu_est = abu_est.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            if dataset == "test":
                target = X.data.get("abd_map_test").permute(1, 2, 0).cpu().numpy()
            elif dataset == "val":
                target = X.data.get("abd_map_val").permute(1, 2, 0).cpu().numpy()
            else:
                target = X.data.get("abd_map").permute(1, 2, 0).cpu().numpy()

            true_endmem = X.data.get("end_mem").numpy()
            est_endmem = endm.detach().cpu().numpy()
            est_endmem = np.average(est_endmem, axis=-1)
            est_endmem = est_endmem.reshape((L, P))
            
            order = []
            for i in range(true_endmem.shape[1]):
                val = 10**5
                idx = 0
                for j in range(est_endmem.shape[1]):
                    if j in order:
                        continue
                    tmp = np.sqrt(np.mean((true_endmem[:, i] - est_endmem[:, j])**2))
                    if tmp < val:
                        val = tmp
                        idx = j
                order.append(idx)
            for j in range(est_endmem.shape[1]):
                if j not in order:
                    order.append(j)
            print(order)

            abu_est = abu_est[:, :, order]
            est_endmem = est_endmem[:, order]

            x = x.view(-1, col, row).permute(1, 2, 0).detach().cpu().numpy()
            re_result = re_result.view(-1, col, row).permute(1, 2, 0).detach().cpu().numpy()
            re = utils.compute_re(x, re_result)
            print("RE:", re)

            rmse_cls, mean_rmse = utils.compute_rmse(target, abu_est)
            print("Class-wise abundance RMSE value:")
            for i in range(len(rmse_cls)):
                print("Class", i + 1, ":", rmse_cls[i])
            print("Mean RMSE:", mean_rmse)

            sad_cls, mean_sad = utils.compute_rmse(true_endmem[:, np.newaxis, :], est_endmem[:, np.newaxis, :])
            print("Class-wise endmember RMSE value:")
            for i in range(len(sad_cls)):
                print("Class", i + 1, ":", sad_cls[i])
            print("Mean RMSE:", mean_sad)

            with open(os.path.dirname(self.out_dir) + "/log.csv", 'a') as file:
                file.write(f"Iter: {iteration}, ")
                file.write(f"DATASET: {dataset}, ")
                file.write(f"PARAMS: 0, ")
                file.write(f"RE: {re:.6f}, ")
                file.write(f"SAD: {mean_sad:.6f}, ")
                file.write(f"RMSE: {mean_rmse:.6f}\n")

            plots.plot_abundance(target, abu_est, P, self.out_dir + "/", name_ext=f"_{dataset}")
            plots.plot_endmembers(true_endmem, est_endmem, P, self.out_dir + "/", name_ext=f"_{dataset}", errors=X.data.get('endm_var').numpy() )
            tot_re += re
            tot_sad += mean_sad
            tot_rmse += mean_rmse
            self.rgb(re_result, dataset, "final", transpose=False) 

        with open(os.path.dirname(self.out_dir) + "/log.csv", 'a') as file:
            file.write(f"Iter: {iteration}, ")
            file.write(f"DATASET: all, ")
            file.write(f"PARAMS: {params}, ")
            file.write(f"RE: {tot_re/3:.6f}, ")
            file.write(f"SAD: {tot_sad/3:.6f}, ")
            file.write(f"RMSE: {tot_rmse/3:.6f}\n")


    def rgb(self, re, epoch, loss, transpose=True):
        if self.WL is None:
            print("No WL provided, skipping RGB creation")
            return True
        # use numpy array as input 
        out_dir = os.path.join(self.out_dir, "rgb")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        re = re.squeeze()
        if transpose:
            re = np.transpose(re, (1, 2, 0))
        img = RGB(re, self.WL)
        for i in range(3):
            img[:, :, i] = (img[:, :, i] - img[:, :, i].min()) / (img[:, :, i].max() - img[:, :, i].min())

        matplotlib.image.imsave(os.path.join(out_dir, f"rgb_{epoch}_{loss}.png"), img)

