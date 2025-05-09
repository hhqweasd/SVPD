import torch
from torch import nn
from torch.nn.modules.utils import _pair
from deformable import DeformConv

class ConicConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=(3,3), bias=False):
        super().__init__()
        self.deform_conv = DeformConv(
            c_in,
            c_out,
            kernel_size=3,
            stride=1,
            padding=(1,1),
            im2col_step=11,
            bias=bias,
            #dilation=kernel_size,
        )
        self.kernel_size = _pair(kernel_size)

    def forward(self, input, vpts):
        N, C, H, W = input.shape
        Kh, Kw = self.kernel_size

        with torch.no_grad():
            ys, xs = torch.meshgrid(
                torch.arange(0, H).float().to(input.device),
                torch.arange(0, W).float().to(input.device),
            )
            # d: [N, H, W, 2]
            d = torch.cat(
                [
                    (vpts[:, 0, None, None] - ys)[..., None],
                    (vpts[:, 1, None, None] - xs)[..., None],
                ],
                dim=-1,
            )
            d /= torch.norm(d, dim=-1, keepdim=True).clamp(min=1e-5)
            n = torch.cat([-d[..., 1:2], d[..., 0:1]], dim=-1)

            offset = torch.zeros((N, H, W, Kh, Kw, 2)).to(input.device)
            for i in range(Kh):
                for j in range(Kw):
                    offset[..., i, j, :] = d * (1 - i) + n * (1 - j)
                    offset[..., i, j, 0] += 1 - i
                    offset[..., i, j, 1] += 1 - j
            offset = offset.permute(0, 3, 4, 5, 1, 2).reshape((N, -1, H, W))
        return self.deform_conv(input, offset)
        
class ConicConv37(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=(3,7), bias=False):
        super().__init__()
        self.deform_conv = DeformConv(
            c_in,
            c_out,
            kernel_size=kernel_size,
            stride=1,
            padding=(1,3),
            im2col_step=11,
            bias=bias,
            #dilation=kernel_size,
        )
        self.kernel_size = _pair(kernel_size)

    def forward(self, input, vpts):
        N, C, H, W = input.shape
        Kh, Kw = self.kernel_size

        with torch.no_grad():
            ys, xs = torch.meshgrid(
                torch.arange(0, H).float().to(input.device),
                torch.arange(0, W).float().to(input.device),
            )
            # d: [N, H, W, 2]
            d = torch.cat(
                [
                    (vpts[:, 0, None, None] - ys)[..., None],
                    (vpts[:, 1, None, None] - xs)[..., None],
                ],
                dim=-1,
            )
            d /= torch.norm(d, dim=-1, keepdim=True).clamp(min=1e-5)
            n = torch.cat([-d[..., 1:2], d[..., 0:1]], dim=-1)

            offset = torch.zeros((N, H, W, Kh, Kw, 2)).to(input.device)
            for i in range(Kh):
                for j in range(Kw):
                    offset[..., i, j, :] = d * (1 - i) + n * (1 - j)
                    offset[..., i, j, 0] += 1 - i
                    offset[..., i, j, 1] += 1 - j
            offset = offset.permute(0, 3, 4, 5, 1, 2).reshape((N, -1, H, W))
        return self.deform_conv(input, offset)

class ConicConv73(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=(7,3), bias=False):
        super().__init__()
        self.deform_conv = DeformConv(
            c_in,
            c_out,
            kernel_size=kernel_size,
            stride=1,
            padding=(3,1),
            im2col_step=11,
            bias=bias,
            #dilation=kernel_size,
        )
        self.kernel_size = _pair(kernel_size)

    def forward(self, input, vpts):
        N, C, H, W = input.shape
        Kh, Kw = self.kernel_size

        with torch.no_grad():
            ys, xs = torch.meshgrid(
                torch.arange(0, H).float().to(input.device),
                torch.arange(0, W).float().to(input.device),
            )
            # d: [N, H, W, 2]
            d = torch.cat(
                [
                    (vpts[:, 0, None, None] - ys)[..., None],
                    (vpts[:, 1, None, None] - xs)[..., None],
                ],
                dim=-1,
            )
            d /= torch.norm(d, dim=-1, keepdim=True).clamp(min=1e-5)
            n = torch.cat([-d[..., 1:2], d[..., 0:1]], dim=-1)

            offset = torch.zeros((N, H, W, Kh, Kw, 2)).to(input.device)
            for i in range(Kh):
                for j in range(Kw):
                    offset[..., i, j, :] = d * (1 - i) + n * (1 - j)
                    offset[..., i, j, 0] += 1 - i
                    offset[..., i, j, 1] += 1 - j
            offset = offset.permute(0, 3, 4, 5, 1, 2).reshape((N, -1, H, W))
        return self.deform_conv(input, offset)          

class ConicConv_test(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=(3,3), bias=False):
        super().__init__()
        self.deform_conv = DeformConv(
            c_in,
            c_out,
            kernel_size=3,
            stride=1,
            padding=(1,1),
            im2col_step=32,
            bias=bias,
            #dilation=kernel_size,
        )
        self.kernel_size = _pair(kernel_size)

    def forward(self, input, vpts):
        N, C, H, W = input.shape
        Kh, Kw = self.kernel_size

        with torch.no_grad():
            ys, xs = torch.meshgrid(
                torch.arange(0, H).float().to(input.device),
                torch.arange(0, W).float().to(input.device),
            )
            # d: [N, H, W, 2]
            d = torch.cat(
                [
                    (vpts[:, 0, None, None] - ys)[..., None],
                    (vpts[:, 1, None, None] - xs)[..., None],
                ],
                dim=-1,
            )
            d /= torch.norm(d, dim=-1, keepdim=True).clamp(min=1e-5)
            n = torch.cat([-d[..., 1:2], d[..., 0:1]], dim=-1)

            offset = torch.zeros((N, H, W, Kh, Kw, 2)).to(input.device)
            for i in range(Kh):
                for j in range(Kw):
                    offset[..., i, j, :] = d * (1 - i) + n * (1 - j)
                    offset[..., i, j, 0] += 1 - i
                    offset[..., i, j, 1] += 1 - j
            offset = offset.permute(0, 3, 4, 5, 1, 2).reshape((N, -1, H, W))
        return self.deform_conv(input, offset)
        
class ConicConv37_test(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=(3,7), bias=False):
        super().__init__()
        self.deform_conv = DeformConv(
            c_in,
            c_out,
            kernel_size=kernel_size,
            stride=1,
            padding=(1,3),
            im2col_step=32,
            bias=bias,
            #dilation=kernel_size,
        )
        self.kernel_size = _pair(kernel_size)

    def forward(self, input, vpts):
        N, C, H, W = input.shape
        Kh, Kw = self.kernel_size

        with torch.no_grad():
            ys, xs = torch.meshgrid(
                torch.arange(0, H).float().to(input.device),
                torch.arange(0, W).float().to(input.device),
            )
            # d: [N, H, W, 2]
            d = torch.cat(
                [
                    (vpts[:, 0, None, None] - ys)[..., None],
                    (vpts[:, 1, None, None] - xs)[..., None],
                ],
                dim=-1,
            )
            d /= torch.norm(d, dim=-1, keepdim=True).clamp(min=1e-5)
            n = torch.cat([-d[..., 1:2], d[..., 0:1]], dim=-1)

            offset = torch.zeros((N, H, W, Kh, Kw, 2)).to(input.device)
            for i in range(Kh):
                for j in range(Kw):
                    offset[..., i, j, :] = d * (1 - i) + n * (1 - j)
                    offset[..., i, j, 0] += 1 - i
                    offset[..., i, j, 1] += 1 - j
            offset = offset.permute(0, 3, 4, 5, 1, 2).reshape((N, -1, H, W))
        return self.deform_conv(input, offset)

class ConicConv73_test(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=(7,3), bias=False):
        super().__init__()
        self.deform_conv = DeformConv(
            c_in,
            c_out,
            kernel_size=kernel_size,
            stride=1,
            padding=(3,1),
            im2col_step=32,
            bias=bias,
            #dilation=kernel_size,
        )
        self.kernel_size = _pair(kernel_size)

    def forward(self, input, vpts):
        N, C, H, W = input.shape
        Kh, Kw = self.kernel_size

        with torch.no_grad():
            ys, xs = torch.meshgrid(
                torch.arange(0, H).float().to(input.device),
                torch.arange(0, W).float().to(input.device),
            )
            # d: [N, H, W, 2]
            d = torch.cat(
                [
                    (vpts[:, 0, None, None] - ys)[..., None],
                    (vpts[:, 1, None, None] - xs)[..., None],
                ],
                dim=-1,
            )
            d /= torch.norm(d, dim=-1, keepdim=True).clamp(min=1e-5)
            n = torch.cat([-d[..., 1:2], d[..., 0:1]], dim=-1)

            offset = torch.zeros((N, H, W, Kh, Kw, 2)).to(input.device)
            for i in range(Kh):
                for j in range(Kw):
                    offset[..., i, j, :] = d * (1 - i) + n * (1 - j)
                    offset[..., i, j, 0] += 1 - i
                    offset[..., i, j, 1] += 1 - j
            offset = offset.permute(0, 3, 4, 5, 1, 2).reshape((N, -1, H, W))
        return self.deform_conv(input, offset)        