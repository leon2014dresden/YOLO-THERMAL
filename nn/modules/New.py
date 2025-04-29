import numpy as np
import torch
from torch import nn
from torch.nn import init
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f
import torch.nn.functional as F
import math
from functools import partial
from typing import Callable, Any

import torch.nn as nn
from einops import rearrange, repeat
from timm.models.layers import DropPath

class GAMAttention(nn.Module):
       #https://paperswithcode.com/paper/global-attention-mechanism-retain-information
    def __init__(self, c1, c2, group=True,rate=4):
        super(GAMAttention, self).__init__()
        
        self.channel_attention = nn.Sequential(
            nn.Linear(c1, int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(c1 / rate), c1)
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(c1, c1//rate, kernel_size=7, padding=3,groups=rate)if group else nn.Conv2d(c1, int(c1 / rate), kernel_size=7, padding=3), 
            nn.BatchNorm2d(int(c1 /rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1//rate, c2, kernel_size=7, padding=3,groups=rate) if group else nn.Conv2d(int(c1 / rate), c2, kernel_size=7, padding=3), 
            nn.BatchNorm2d(c2)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        x = x * x_channel_att
 
        x_spatial_att = self.spatial_attention(x).sigmoid()
        x_spatial_att=channel_shuffle(x_spatial_att,4) #last shuffle 
        out = x * x_spatial_att
        return out  

def channel_shuffle(x, groups=2):
        B, C, H, W = x.size()
        out = x.view(B, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous()
        out=out.view(B, C, H, W) 
        return out
    

class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
 
    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class BasicConv(nn.Module):   #https://arxiv.org/pdf/2010.03045.pdf
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
 
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
 
 
class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
 
 
class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
 
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale
 
 
class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()
 
    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out


class SPPFCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=5):
        super(SPPFCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        x2 = self.m(x1)
        x3 = self.m(x2)
        y1 = self.cv6(self.cv5(torch.cat((x1,x2,x3, self.m(x3)),1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))
    

import torch
import torch.nn as nn
from einops import rearrange


class FocusedLinearAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size=[20, 20], num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 focusing_factor=3, kernel_size=5):

        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dim = int(dim)
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.focusing_factor = focusing_factor
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.window_size = window_size
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, window_size[0] * window_size[1], dim)))

        self.softmax = nn.Softmax(dim=-1)

        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv.unbind(0)
        k = k + self.positional_encoding[:, :k.shape[1], :]
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        scale = nn.Softplus()(self.scale)
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        if float(focusing_factor) <= 6:
            q = q ** focusing_factor
            k = k ** focusing_factor
        else:
            q = (q / q.max(dim=-1, keepdim=True)[0]) ** focusing_factor
            k = (k / k.max(dim=-1, keepdim=True)[0]) ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
        if i * j * (c + d) > c * d * (i + j):
            kv = torch.einsum("b j c, b j d -> b c d", k, v)
            x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        else:
            qk = torch.einsum("b i c, b j c -> b i j", q, k)
            x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)

        num = int(v.shape[1] ** 0.5)
        feature_map = rearrange(v, "b (w h) c -> b c w h", w=num, h=num)
        feature_map = rearrange(self.dwc(feature_map), "b c w h -> b (w h) c")
        x = x + feature_map
        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, "b (w h) c -> b c w h", b=B, c=self.dim, w=num, h=num)
        return x


class DySnakeConv(nn.Module):
    def __init__(self, inc, ouc, k=3, act=True) -> None:
        super().__init__()

        self.conv_0 = Conv(inc, ouc, k, act=act)
        self.conv_x = DSConv(inc, ouc, 0, k, act=True)
        self.conv_y = DSConv(inc, ouc, 1, k, act=True)
        self.conv_1x1 = Conv(ouc * 3, ouc, 1, act=act)

    def forward(self, x):
        return self.conv_1x1(torch.cat([self.conv_0(x), self.conv_x(x), self.conv_y(x)], dim=1))


class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, morph, kernel_size=3, if_offset=True, extend_scope=1, act=True):
        """
        The Dynamic Snake Convolution
        :param in_ch: input channel
        :param out_ch: output channel
        :param kernel_size: the size of kernel
        :param extend_scope: the range to expand (default 1 for this method)
        :param morph: the morphology of the convolution kernel is mainly divided into two types
                        along the x-axis (0) and the y-axis (1) (see the paper for details)
        :param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
        """
        super(DSConv, self).__init__()
        # use the <offset_conv> to learn the deformable offset
        self.offset_conv = nn.Conv2d(in_ch, 2 * kernel_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(2 * kernel_size)
        self.kernel_size = kernel_size

        # two types of the DSConv (along x-axis and y-axis)
        if morph == 0:
            self.dsc_conv_x = nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=(kernel_size, 1),
                stride=(kernel_size, 1),
                padding=0,
            )
        else:
            self.dsc_conv_y = nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=(1, kernel_size),
                stride=(1, kernel_size),
                padding=0,
            )

        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset

    def forward(self, f):
        offset = self.offset_conv(f)
        offset = self.bn(offset)
        # We need a range of deformation between -1 and 1 to mimic the snake's swing
        offset = torch.tanh(offset)
        input_shape = f.shape
        dsc = DSC(input_shape, self.kernel_size, self.extend_scope, self.morph)
        deformed_feature = dsc.deform_conv(f, offset, self.if_offset)
        if self.morph == 0:
            x = self.dsc_conv_x(deformed_feature.type(f.dtype))
            x = self.gn(x)
            x = self.act(x)
            return x
        else:
            x = self.dsc_conv_y(deformed_feature.type(f.dtype))
            x = self.gn(x)
            x = self.act(x)
            return x


# Core code, for ease of understanding, we mark the dimensions of input and output next to the code
class DSC(object):
    def __init__(self, input_shape, kernel_size, extend_scope, morph):
        self.num_points = kernel_size
        self.width = input_shape[2]
        self.height = input_shape[3]
        self.morph = morph
        self.extend_scope = extend_scope  # offset (-1 ~ 1) * extend_scope

        # define feature map shape
        """
        B: Batch size  C: Channel  W: Width  H: Height
        """
        self.num_batch = input_shape[0]
        self.num_channels = input_shape[1]

    """
    input: offset [B,2*K,W,H]  K: Kernel size (2*K: 2D image, deformation contains <x_offset> and <y_offset>)
    output_x: [B,1,W,K*H]   coordinate map
    output_y: [B,1,K*W,H]   coordinate map
    """

    def _coordinate_map_3D(self, offset, if_offset):
        device = offset.device
        # offset
        y_offset, x_offset = torch.split(offset, self.num_points, dim=1)

        y_center = torch.arange(0, self.width).repeat([self.height])
        y_center = y_center.reshape(self.height, self.width)
        y_center = y_center.permute(1, 0)
        y_center = y_center.reshape([-1, self.width, self.height])
        y_center = y_center.repeat([self.num_points, 1, 1]).float()
        y_center = y_center.unsqueeze(0)

        x_center = torch.arange(0, self.height).repeat([self.width])
        x_center = x_center.reshape(self.width, self.height)
        x_center = x_center.permute(0, 1)
        x_center = x_center.reshape([-1, self.width, self.height])
        x_center = x_center.repeat([self.num_points, 1, 1]).float()
        x_center = x_center.unsqueeze(0)

        if self.morph == 0:
            """
            Initialize the kernel and flatten the kernel
                y: only need 0
                x: -num_points//2 ~ num_points//2 (Determined by the kernel size)
                !!! The related PPT will be submitted later, and the PPT will contain the whole changes of each step
            """
            y = torch.linspace(0, 0, 1)
            x = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )

            y, x = torch.meshgrid(y, x)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)  # [B*K*K, W,H]

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)  # [B*K*K, W,H]

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1).to(device)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1).to(device)

            y_offset_new = y_offset.detach().clone()

            if if_offset:
                y_offset = y_offset.permute(1, 0, 2, 3)
                y_offset_new = y_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)

                # The center position remains unchanged and the rest of the positions begin to swing
                # This part is quite simple. The main idea is that "offset is an iterative process"
                y_offset_new[center] = 0
                for index in range(1, center):
                    y_offset_new[center + index] = (y_offset_new[center + index - 1] + y_offset[center + index])
                    y_offset_new[center - index] = (y_offset_new[center - index + 1] + y_offset[center - index])
                y_offset_new = y_offset_new.permute(1, 0, 2, 3).to(device)
                y_new = y_new.add(y_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height])
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape([
                self.num_batch, self.num_points * self.width, 1 * self.height
            ])
            x_new = x_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height])
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape([
                self.num_batch, self.num_points * self.width, 1 * self.height
            ])
            return y_new, x_new

        else:
            """
            Initialize the kernel and flatten the kernel
                y: -num_points//2 ~ num_points//2 (Determined by the kernel size)
                x: only need 0
            """
            y = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )
            x = torch.linspace(0, 0, 1)

            y, x = torch.meshgrid(y, x)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1)

            y_new = y_new.to(device)
            x_new = x_new.to(device)
            x_offset_new = x_offset.detach().clone()

            if if_offset:
                x_offset = x_offset.permute(1, 0, 2, 3)
                x_offset_new = x_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)
                x_offset_new[center] = 0
                for index in range(1, center):
                    x_offset_new[center + index] = (x_offset_new[center + index - 1] + x_offset[center + index])
                    x_offset_new[center - index] = (x_offset_new[center - index + 1] + x_offset[center - index])
                x_offset_new = x_offset_new.permute(1, 0, 2, 3).to(device)
                x_new = x_new.add(x_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height])
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape([
                self.num_batch, 1 * self.width, self.num_points * self.height
            ])
            x_new = x_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height])
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape([
                self.num_batch, 1 * self.width, self.num_points * self.height
            ])
            return y_new, x_new

    """
    input: input feature map [N,C,D,W,H]；coordinate map [N,K*D,K*W,K*H] 
    output: [N,1,K*D,K*W,K*H]  deformed feature map
    """

    def _bilinear_interpolate_3D(self, input_feature, y, x):
        device = input_feature.device
        y = y.reshape([-1]).float()
        x = x.reshape([-1]).float()

        zero = torch.zeros([]).int()
        max_y = self.width - 1
        max_x = self.height - 1

        # find 8 grid locations
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)

        input_feature_flat = input_feature.flatten()
        input_feature_flat = input_feature_flat.reshape(
            self.num_batch, self.num_channels, self.width, self.height)
        input_feature_flat = input_feature_flat.permute(0, 2, 3, 1)
        input_feature_flat = input_feature_flat.reshape(-1, self.num_channels)
        dimension = self.height * self.width

        base = torch.arange(self.num_batch) * dimension
        base = base.reshape([-1, 1]).float()

        repeat = torch.ones([self.num_points * self.width * self.height
                             ]).unsqueeze(0)
        repeat = repeat.float()

        base = torch.matmul(base, repeat)
        base = base.reshape([-1])

        base = base.to(device)

        base_y0 = base + y0 * self.height
        base_y1 = base + y1 * self.height

        # top rectangle of the neighbourhood volume
        index_a0 = base_y0 - base + x0
        index_c0 = base_y0 - base + x1

        # bottom rectangle of the neighbourhood volume
        index_a1 = base_y1 - base + x0
        index_c1 = base_y1 - base + x1

        # get 8 grid values
        value_a0 = input_feature_flat[index_a0.type(torch.int64)].to(device)
        value_c0 = input_feature_flat[index_c0.type(torch.int64)].to(device)
        value_a1 = input_feature_flat[index_a1.type(torch.int64)].to(device)
        value_c1 = input_feature_flat[index_c1.type(torch.int64)].to(device)

        # find 8 grid locations
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume
        y0 = torch.clamp(y0, zero, max_y + 1)
        y1 = torch.clamp(y1, zero, max_y + 1)
        x0 = torch.clamp(x0, zero, max_x + 1)
        x1 = torch.clamp(x1, zero, max_x + 1)

        x0_float = x0.float()
        x1_float = x1.float()
        y0_float = y0.float()
        y1_float = y1.float()

        vol_a0 = ((y1_float - y) * (x1_float - x)).unsqueeze(-1).to(device)
        vol_c0 = ((y1_float - y) * (x - x0_float)).unsqueeze(-1).to(device)
        vol_a1 = ((y - y0_float) * (x1_float - x)).unsqueeze(-1).to(device)
        vol_c1 = ((y - y0_float) * (x - x0_float)).unsqueeze(-1).to(device)

        outputs = (value_a0 * vol_a0 + value_c0 * vol_c0 + value_a1 * vol_a1 +
                   value_c1 * vol_c1)

        if self.morph == 0:
            outputs = outputs.reshape([
                self.num_batch,
                self.num_points * self.width,
                1 * self.height,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
        else:
            outputs = outputs.reshape([
                self.num_batch,
                1 * self.width,
                self.num_points * self.height,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
        return outputs

    def deform_conv(self, input, offset, if_offset):
        y, x = self._coordinate_map_3D(offset, if_offset)
        deformed_feature = self._bilinear_interpolate_3D(input, y, x)
        return deformed_feature
    

class Bottleneck_DySnakeConv(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DySnakeConv(c1, c_)
        self.cv2 = DySnakeConv(c_, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    
    
class C2f_DySnakeConv(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_DySnakeConv(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, c1, c2, dimension=1):
        super().__init__()
        self.d = dimension
        self.cv1 = Conv(c1*4, c2)
        
    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = self.cv1(x)
        return x
        

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            Conv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        )

        # carafe
        # from mmcv.ops import CARAFEPack
        # self.upsample = nn.Sequential(
        #     BasicConv(in_channels, out_channels, 1),
        #     CARAFEPack(out_channels, scale_factor=scale_factor)
        # )

    def forward(self, x):
        x = self.upsample(x)

        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Downsample, self).__init__()

        self.downsample = nn.Sequential(
            Conv(in_channels, out_channels, scale_factor, scale_factor, 0)
        )

    def forward(self, x):
        x = self.downsample(x)

        return x


class ASFF_2(nn.Module):
    def __init__(self, inter_dim=512, level=0, channel=[256, 512]):
        super(ASFF_2, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)

        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)
        self.upsample = Upsample(channel[1], channel[0])
        self.downsample = Downsample(channel[0], channel[1])
        self.level = level
        self.c2f = C2f(inter_dim, inter_dim)
    def forward(self, x):
        input1, input2 = x
        if self.level == 0:
            input2 = self.upsample(input2)
        elif self.level == 1:
            input1 = self.downsample(input1)

        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :]

        out = self.conv(fused_out_reduced)

        return self.c2f(out)


class ASFF_3(nn.Module):
    def __init__(self, inter_dim=512, level=0, channel=[64, 128, 256]):
        super(ASFF_3, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)

        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)

        self.level = level
        if self.level == 0:
            self.upsample4x = Upsample(channel[2], channel[0], scale_factor=4)
            self.upsample2x = Upsample(channel[1], channel[0], scale_factor=2)
        elif self.level == 1:
            self.upsample2x1 = Upsample(channel[2], channel[1], scale_factor=2)
            self.downsample2x1 = Downsample(channel[0], channel[1], scale_factor=2)
        elif self.level == 2:
            self.downsample2x = Downsample(channel[1], channel[2], scale_factor=2)
            self.downsample4x = Downsample(channel[0], channel[2], scale_factor=4)
        self.c2f = C2f(inter_dim, inter_dim)

    def forward(self, x):
        input1, input2, input3 = x
        if self.level == 0:
            input2 = self.upsample2x(input2)
            input3 = self.upsample4x(input3)
        elif self.level == 1:
            input3 = self.upsample2x1(input3)
            input1 = self.downsample2x1(input1)
        elif self.level == 2:
            input1 = self.downsample4x(input1)
            input2 = self.downsample2x(input2)
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        level_3_weight_v = self.weight_level_3(input3)

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :] + \
                            input3 * levels_weight[:, 2:, :, :]

        out = self.conv(fused_out_reduced)

        return self.c2f(out)


import torch
import math
from functools import partial
from typing import Callable, Any

import torch.nn as nn
from einops import rearrange, repeat
from timm.models.layers import DropPath

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
try:
    import selective_scan_cuda_core
    import selective_scan_cuda_oflex
    import selective_scan_cuda_ndstate
    import selective_scan_cuda_nrow
    import selective_scan_cuda
except:
    pass

try:
    "sscore acts the same as mamba_ssm"
    import selective_scan_cuda_core
except Exception as e:
    print(e, flush=True)
    "you should install mamba_ssm to use this"
    SSMODE = "mamba_ssm"
    import selective_scan_cuda
    # from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref


class LayerNorm2d(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        return x


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


# Cross Scan
class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs, None, None


# cross selective scan ===============================
class SelectiveScanCore(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        # all in float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None and D.stride(-1) != 1:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True
        ctx.delta_softplus = delta_softplus
        ctx.backnrows = backnrows
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


def cross_selective_scan(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        out_norm: torch.nn.Module = None,
        out_norm_shape="v0",
        nrows=-1,  # for SelectiveScanNRow
        backnrows=-1,  # for SelectiveScanNRow
        delta_softplus=True,
        to_dtype=True,
        force_fp32=False,  # False if ssoflex
        ssoflex=True,
        SelectiveScan=None,
        scan_mode_type='default'
):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

    xs = CrossScan.apply(x)

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    # HiPPO matrix
    As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)  # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
    ).view(B, K, -1, H, W)

    y: torch.Tensor = CrossMerge.apply(ys)

    if out_norm_shape in ["v1"]:  # (B, C, H, W)
        y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1)  # (B, H, W, C)
    else:  # (B, L, C)
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)


class SS2D(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # ======================
            forward_type="v2",
            **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv
        self.K = 4

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        self.out_norm = nn.LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v2=partial(self.forward_corev2, force_fp32=None, SelectiveScan=SelectiveScanCore),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, FORWARD_TYPES.get("v2", None))

        # in proj =======================================
        d_proj = d_expand if self.disable_z else (d_expand * 2)
        self.in_proj = nn.Conv2d(d_model, d_proj, kernel_size=1, stride=1, groups=1, bias=bias, **factory_kwargs)
        self.act: nn.Module = nn.GELU()

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False,
                      **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_proj = nn.Conv2d(d_expand, d_model, kernel_size=1, stride=1, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # simple init dt_projs, A_logs, Ds
        self.Ds = nn.Parameter(torch.ones((self.K * d_inner)))
        self.A_logs = nn.Parameter(
            torch.zeros((self.K * d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
        self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
        self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev2(self, x: torch.Tensor, channel_first=False, SelectiveScan=SelectiveScanCore,
                       cross_selective_scan=cross_selective_scan, force_fp32=None):
        force_fp32 = (self.training and (not self.disable_force32)) if force_fp32 is None else force_fp32
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        x = cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds,
            out_norm=getattr(self, "out_norm", None),
            out_norm_shape=getattr(self, "out_norm_shape", "v0"),
            delta_softplus=True, force_fp32=force_fp32,
            SelectiveScan=SelectiveScan, ssoflex=self.training,  # output fp32
        )
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=1)  # (b, d, h, w)
            if not self.disable_z_act:
                z1 = self.act(z)
        if self.d_conv > 0:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x, channel_first=(self.d_conv > 1))
        y = y.permute(0, 3, 1, 2).contiguous()
        if not self.disable_z:
            y = y * z1
        out = self.dropout(self.out_proj(y))
        return out


class RGBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True,
                                groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x) + x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LSBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0):
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=3, padding=3 // 2, groups=hidden_features)
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, padding=0)
        self.act = act_layer()
        self.fc3 = nn.Conv2d(hidden_features, in_features, kernel_size=1, padding=0)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        input = x
        x = self.fc1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = input + self.drop(x)
        return x


class XSSBlock(nn.Module):
    def __init__(
            self,
            in_channels: int = 0,
            hidden_dim: int = 0,
            n: int = 1,
            mlp_ratio=4.0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        ) if in_channels != hidden_dim else nn.Identity()
        self.hidden_dim = hidden_dim
        # ==========SSM============================
        self.norm = norm_layer(hidden_dim)
        self.ss2d = nn.Sequential(*(SS2D(d_model=self.hidden_dim,
                                         d_state=ssm_d_state,
                                         ssm_ratio=ssm_ratio,
                                         ssm_rank_ratio=ssm_rank_ratio,
                                         dt_rank=ssm_dt_rank,
                                         act_layer=ssm_act_layer,
                                         d_conv=ssm_conv,
                                         conv_bias=ssm_conv_bias,
                                         dropout=ssm_drop_rate, ) for _ in range(n)))
        self.drop_path = DropPath(drop_path)
        self.lsblock = LSBlock(hidden_dim, hidden_dim)
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = RGBlock(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                               drop=mlp_drop_rate)

    def forward(self, input):
        input = self.in_proj(input)
        # ====================
        X1 = self.lsblock(input)
        input = input + self.drop_path(self.ss2d(self.norm(X1)))
        # ===================
        if self.mlp_branch:
            input = input + self.drop_path(self.mlp(self.norm2(input)))
        return input


class VSSBlock(nn.Module):
    def __init__(
            self,
            in_channels: int = 0,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        # proj
        self.proj_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
            )

        self.drop_path = DropPath(drop_path)
        self.lsblock = LSBlock(hidden_dim, hidden_dim)
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = RGBlock(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                               drop=mlp_drop_rate, channels_first=False)

    def forward(self, input: torch.Tensor):
        input = self.proj_conv(input)
        X1 = self.lsblock(input)
        x = input + self.drop_path(self.op(self.norm(X1)))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x


class SimpleStem(nn.Module):
    def __init__(self, inp, embed_dim, ks=3):
        super().__init__()
        self.hidden_dims = embed_dim // 2
        self.conv = nn.Sequential(
            nn.Conv2d(inp, self.hidden_dims, kernel_size=ks, stride=2, padding=autopad(ks, d=1), bias=False),
            nn.BatchNorm2d(self.hidden_dims),
            nn.GELU(),
            nn.Conv2d(self.hidden_dims, embed_dim, kernel_size=ks, stride=2, padding=autopad(ks, d=1), bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.conv(x)


class VisionClueMerge(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.hidden = int(dim * 4)

        self.pw_linear = nn.Sequential(
            nn.Conv2d(self.hidden, out_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim),
            nn.SiLU()
        )

    def forward(self, x):
        y = torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
        ], dim=1)
        return self.pw_linear(y)
    
    



if __name__ == '__main__':
    # X = torch.randn(1, 256, 16, 16)
    # model = SPPFCSPC(256, 256)
    # print(model(X).shape)
    
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # input = torch.randn(1, 128, 20, 20)
    # siatt = FocusedLinearAttention(dim=128).to(device)
    # output = siatt(input.to(device))
    # print(output.shape)
    
    # input = torch.randn(1, 128, 8, 8)
    # dsconv = DySnakeConv(128, 256)
    # output = dsconv(input)
    # print(output.shape)

    # input = torch.randn(1, 128, 8, 8)
    # spdconv = SPDConv(128, 256)
    # output = spdconv(input)
    # print(output.shape)
    
    
    input_0 = torch.randn(1, 256, 40, 40)
    input_1 = torch.randn(1, 512, 20, 20)
    input = [input_0, input_1]
    model1 = ASFF_2(256, level=0, channel=[256, 512])
    output = model1(input)
    print(output.shape)

    input_0 = torch.randn(1, 256, 80, 80)
    input_1 = torch.randn(1, 512, 40, 40)
    input_2 = torch.randn(1, 1024, 20, 20)
    input = [input_0, input_1, input_2]
    model2 = ASFF_3(1024, level=2, channel=[256, 512, 1024])
    output = model2(input)
    print(output.shape)