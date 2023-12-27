
import torch
from torch import nn
import torch.nn.functional as TF


from math import e, pi


def kernel_weights_Gausian(l, sigma):
    fw = torch.zeros((l, l)).float()

    for x in range(l//2+1):
        for y in range(l // 2+1):

            d2 = (x*x + y*y)/2*sigma
            A = (1 - d2 / 2 * (sigma ** 2))
            B = ((e ** -d2) / (pi * (sigma ** 4)))
            v = B*A

            fw[l // 2 - x, l // 2 - y] = v
            fw[l // 2 - x, l // 2 + y] = v
            fw[l // 2 + x, l // 2 - y] = v
            fw[l // 2 + x, l // 2 + y] = v
    fw = torch.unsqueeze(fw, 0)
    fw = torch.unsqueeze(fw, 0)
    fw = torch.mul(fw, 1 / fw.sum())
    return fw


def kernel_weights_Sharpen():
    fw = torch.zeros((3, 3)).float()
    fw[0, 1] = -1
    fw[1, 0] = -1
    fw[1, 2] = -1
    fw[2, 1] = -1
    fw[1, 1] = 5

    fw = torch.unsqueeze(fw, 0)
    fw = torch.unsqueeze(fw, 0)

    return fw


def kernel_weights_DIF(t_conv_kernel):
    if t_conv_kernel == 2:
        tw = torch.tensor([-1.0, 1.0])
    elif t_conv_kernel == 3:
        tw = torch.tensor([-0.5, 1.0, -0.5])
    elif t_conv_kernel == 4:
        tw = torch.tensor([0.25, -0.75, 0.75, -0.25])
    elif t_conv_kernel == 5:
        tw = torch.tensor([-0.125, 0.5, -0.75, 0.5, -0.125])

    else:
        print("ERROR: undefined time conv1d size:", t_conv_kernel)

    tw = torch.unsqueeze(tw, 0)
    tw = torch.unsqueeze(tw, 0)
    return tw


def kernel_weights_grey():
    trgb = torch.tensor([1 / 3, 1 / 3, 1 / 3])
    trgb = torch.unsqueeze(trgb, 0)
    trgb = torch.unsqueeze(trgb, 0)
    return trgb


class MDFM(torch.nn.Module):

    def __init__(self, param_motion):
        super(MDFM, self).__init__()

        self.sharpen_cycles = param_motion["sharpen_cycles"]
        self.HW_conv_kernel = param_motion["HW_conv_kernel"]
        self.HW_conv_sigma = param_motion["HW_conv_sigma"]

        k_frames = param_motion["k_frames"]
        self.k_frames = k_frames
        self.t_conv_kernel = k_frames - 2

        self.norm_add = param_motion["normadd"]

        # self.MDM_conv_sharp = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        # self.MDM_conv_sharp.weight = torch.nn.Parameter(kernel_weights_Sharpen())

        self.MDM_conv_HW = nn.Conv2d(
            1, 1, kernel_size=self.HW_conv_kernel,  padding=self.HW_conv_kernel//2, bias=False)  # ,
        self.MDM_conv_HW.weight = torch.nn.Parameter(
            kernel_weights_Gausian(self.HW_conv_kernel, self.HW_conv_sigma))

        self.MDM_conv_T = nn.Conv1d(
            1, 1, kernel_size=self.t_conv_kernel, bias=False)
        self.MDM_conv_T.weight = torch.nn.Parameter(
            kernel_weights_DIF(self.t_conv_kernel))

        self.MDM_conv_RGB = nn.Conv1d(1, 1, kernel_size=3, bias=False)
        self.MDM_conv_RGB.weight = torch.nn.Parameter(kernel_weights_grey())

    def forward(self, X):
        with torch.no_grad():
            # batch size, n -segement sampled from video , t - consecutive frames, c - RGB , Height,Width
            [b, m, n, t, c, h, w] = X.size()
            # print("X.size()", X.size())

            # X = self.MDM_conv_HW(X.view((-1, 1, h, w)))
            # X = X.view((b, m, n, t, c, h, w))

            X = X.permute((0, 1, 2, 4, 5, 6, 3))
            X = X.reshape(-1, 1, X.size()[-1])
            X = self.MDM_conv_T(X)
            X = X.reshape(b, m, n, c, h, w, t - self.t_conv_kernel + 1)
            # print("X_out.size()", X_out.size())

            # X_out = X_out.permute((0, 1, 5,  3, 4, 2) )
            X = X.permute((0, 1, 2, 6, 4, 5, 3))
            # print("X_out.size()", X_out.size())

            X = X.reshape(-1, 1, X.size()[-1])
            X = self.MDM_conv_RGB(X)

            # X = X.reshape(-1, t - self.t_conv_kernel + 1, h, w)

            for s in range(self.sharpen_cycles):
                X = self.MDM_conv_HW(X.view((-1, 1, h, w)))

            X = X.reshape(b, m, n, t - self.t_conv_kernel + 1, h, w)
            X = torch.add(X, self.norm_add)

        return X
