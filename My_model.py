# Copyright 2023 Sea Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
DCAH-Net implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import math
import numpy as np
from torch import Tensor
from torch.jit.annotations import List
from torch.utils import checkpoint
import torch.utils.checkpoint as cp
from collections import OrderedDict


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .96, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'volo': _cfg(crop_pct=0.96),
    'volo_large': _cfg(crop_pct=1.15),
}

class AssistantLayer(nn.Module):
    def __init__(self, dim, drop=0.3, assist_cls=3):
        super().__init__()

        self.act = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(dim, assist_cls)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class Assistant_Layer_After_Trans(nn.Module):
    def __init__(self, dim, drop=0.3, assist_cls=3):
        super().__init__()

        self.act = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(dim, assist_cls)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False, norm_layer=None):
        super(_DenseLayer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.add_module('norm1', norm_layer(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        # self.add_module('relu1', nn.LeakyReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', norm_layer(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        # self.add_module('relu2', nn.LeakyReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        # type: (List[Tensor]) -> Tensor
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (List[Tensor]) -> (Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (Tensor) -> (Tensor)
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, x):
        return checkpoint.checkpoint(self.forward_function, x)

    def forward_function(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, norm_layer=None):
        super(_Transition, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.add_module('norm', norm_layer(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        # self.add_module('relu', nn.LeakyReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class IntensiveBlock(nn.Module):
    """Implementation of IntensiveBlock"""

    # def __init__(self, dim, num_heads, hidden_dim=64, stem_stride=1, times=1, drop=0.3):
    def __init__(self, dim, hidden_dim=64, out_dim=None, stem_stride=1, times=6, drop=0.3):
        super().__init__()
        # This is first layer used to compress the channels.

        # head_dim = dim // num_heads
        # self.scale = head_dim ** -0.5
        if out_dim is None:
            out_dim = dim
        self.repeat = times

        self.Denseblock = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, hidden_dim, kernel_size=1, stride=stem_stride,
                      padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=3, stride=stem_stride,
                      padding=1, bias=False)
        )

        self.Concat = torch.cat
        # self.Liner_1 = nn.Linear(2 * hidden_dim, dim)
        # self.norm = nn.LayerNorm(4 * hidden_dim)
        # self.act = nn.ReLU(inplace=True)
        # self.Liner_2 = nn.Linear(dim, dim)
        # Place for dropout layer
        # self.drop = nn.Dropout(drop)

    def forward(self, x):
        # x = x.permute(0, 3, 1, 2)
        feature = x
        for i in range(self.repeat):
            feature = self.Denseblock(feature)

        x = self.Concat((feature, x), dim=1)
        # x = x.permute(0, 2, 3, 1)
        return x


class TransLayer(nn.Module):
    def __init__(self, dim, hidden_dim=None, conv_kernel_size=1, conv_stride=1, con_padding=0,
                 pool_kernel_size=2, drop_path=0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim
        self.transLayer = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, hidden_dim, kernel_size=conv_kernel_size, stride=conv_stride,
                      padding=con_padding, bias=False),
            nn.AvgPool2d(kernel_size=pool_kernel_size, stride=2),
        )

    def forward(self, x):
        # x = x.permute(0, 3, 1, 2)
        x = self.transLayer(x)
        # x = x.permute(0, 2, 3, 1)
        return x


class IntensiveAttention(nn.Module):
    # def __init__(self, dim, num_heads=1, mlp_ratio=3,
    #              norm_layer=nn.LayerNorm, act_layer=nn.GELU,
    #              drop_path=0):
    def __init__(self, dim, hidden_dim=[128, 128], drop_path=0):
        super().__init__()
        # self.norm = norm_layer(dim)
        self.block = nn.Sequential(
            IntensiveBlock(dim, hidden_dim[0]),
            TransLayer(hidden_dim[1] + dim, hidden_dim[0]),
            IntensiveBlock(hidden_dim[0], hidden_dim[1]),
            # nn.AvgPool2d(kernel_size=2, stride=2)
            # nn.AdaptiveAvgPool2d((1, 1))
        )
        # self.attn = IntensiveBlock(dim, num_heads)
        # mlp_hidden_dim = mlp_ratio * dim
        # self.mlp = Mlp(in_features=dim,
        #                hidden_features=mlp_hidden_dim,
        #                act_layer=act_layer)
        self.globalpooling = nn.AdaptiveAvgPool2d((1, 1))
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # x = x + self.drop_path(self.attn(self.norm(x)))
        # x = x + self.drop_path(self.mlp(self.norm(x)))
        x = self.drop_path(self.block(x))
        x = x.permute(0, 3, 1, 2)
        x = self.globalpooling(x)

        return x


class OutlookAttention(nn.Module):
    """
    Implementation of outlook attention
    --dim: hidden dim
    --num_heads: number of heads
    --kernel_size: kernel size in each window for outlook attention
    return: token features after outlook attention
    """

    def __init__(self, dim, num_heads, kernel_size=3, padding=1, stride=1,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = qk_scale or head_dim ** -0.5

        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        # self.attn = nn.Linear(dim, kernel_size ** 4 * num_heads)
        self.attn_computation = nn.Conv2d(dim, kernel_size ** 4 * num_heads, kernel_size=1, stride=1, padding=0,
                                          bias=False)
        # self.attn_1 = nn.Linear(dim, 64)
        # self.attn_2 = nn.Linear(64, kernel_size ** 4 * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

    def forward(self, x):
        B, H, W, C = x.shape

        v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W

        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        v = self.unfold(v).reshape(B, self.num_heads, C // self.num_heads,
                                   self.kernel_size * self.kernel_size,
                                   h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/head

        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # Special-attention-1
        # attn = self.attn_1(attn)
        # attn = self.attn_2(attn)
        # attn = self.attn(attn)
        attn = self.attn_computation(attn.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        attn = attn.reshape(
            B, h * w, self.num_heads, self.kernel_size * self.kernel_size,
            self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)

        # Special-attention-2
        # attn = self.attn_computation(attn).reshape(
        #     B, h * w, self.num_heads, self.kernel_size * self.kernel_size,
        #     self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)

        # Original-attention
        # attn = self.attn(attn).reshape(
        #     B, h * w, self.num_heads, self.kernel_size * self.kernel_size,
        #        self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk

        # Neighbour-attention
        # ----------------------------------------------------------------------------------------------
        # Neighbour = attn.mean(dim=-3, keepdim=True)
        # Neighbour = Neighbour.repeat(self.padding, self.padding, h * w, self.padding, self.padding)
        # attn = torch.mul(Neighbour, attn)
        # attn = attn.mean(dim=-2, keepdim=True).mean(dim=-1, keepdim=True).repeat(self.padding, self.padding,
        #                                                                          self.padding,
        #                                                                          self.kernel_size * self.kernel_size,
        #                                                                          self.kernel_size * self.kernel_size)
        # attn = attn.mean(dim=-2, keepdim=True).repeat(self.padding, self.padding,
        #                                               self.padding,
        #                                               self.kernel_size * self.kernel_size,
        #                                               self.padding)
        # ----------------------------------------------------------------------------------------------
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
            B, C * self.kernel_size * self.kernel_size, h * w)
        x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size,
                   padding=self.padding, stride=self.stride)

        x = self.proj(x.permute(0, 2, 3, 1))
        x = self.proj_drop(x)

        return x


class Outlooker(nn.Module):
    """
    Implementation of outlooker layer: which includes outlook attention + MLP
    Outlooker is the first stage in our VOLO
    --dim: hidden dim
    --num_heads: number of heads
    --mlp_ratio: mlp ratio
    --kernel_size: kernel size in each window for outlook attention
    return: outlooker layer
    """

    def __init__(self, dim, kernel_size, padding, stride=1,
                 num_heads=1, mlp_ratio=3., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, qkv_bias=False,
                 qk_scale=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = OutlookAttention(dim, num_heads, kernel_size=kernel_size,
                                     padding=padding, stride=stride,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     attn_drop=attn_drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Mlp(nn.Module):
    """Implementation of MLP"""

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Implementation of self-attention"""

    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape

        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # ------------------------------------------------------------------
        # hadamard_product = q * k * self.scale

        # if self.stride > 1:
        #     hadamard_product = F.avg_pool2d(hadamard_product, self.stride)
        # ------------------------------------------------------------------
        # attn = hadamard_product.softmax(dim=-1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # x = (attn * v).transpose(1, 2).reshape(B, H, W, C)
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Transformer(nn.Module):
    """
    Implementation of Transformer,
    Transformer is the second stage in our VOLO
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop)

        # self.shift_attn = ShiftViTBlock(dim, mlp_ratio=mlp_ratio, drop=attn_drop,
        #                                 drop_path=drop_path, act_layer=act_layer,
        #                                 norm_layer=norm_layer)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # shift
        # x = x + self.drop_path(self.shift_attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ClassAttention(nn.Module):
    """
    Class attention layer from CaiT, see details in CaiT
    Class attention is the post stage in our VOLO, which is optional.
    """

    def __init__(self, dim, num_heads=8, head_dim=None, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            head_dim = dim // num_heads
            self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.kv = nn.Linear(dim,
                            self.head_dim * self.num_heads * 2,
                            bias=qkv_bias)
        self.q = nn.Linear(dim, self.head_dim * self.num_heads, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        kv = self.kv(x).reshape(B, N, 2, self.num_heads,
                                self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[
            1]  # make torchscript happy (cannot use tensor as tuple)
        q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, self.head_dim)
        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        cls_embed = (attn @ v).transpose(1, 2).reshape(
            B, 1, self.head_dim * self.num_heads)
        cls_embed = self.proj(cls_embed)
        cls_embed = self.proj_drop(cls_embed)
        return cls_embed


class ClassBlock(nn.Module):
    """
    Class attention block from CaiT, see details in CaiT
    We use two-layers class attention in our VOLO, which is optional.
    """

    def __init__(self, dim, num_heads, head_dim=None, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ClassAttention(
            dim, num_heads=num_heads, head_dim=head_dim, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x):
        cls_embed = x[:, :1]
        cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
        cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
        return torch.cat([cls_embed, x[:, 1:]], dim=1)


def get_block(block_type, **kargs):
    """
    get block by name, specifically for class attention block in here
    """
    if block_type == 'ca':
        return ClassBlock(**kargs)


def rand_bbox(size, lam, scale=1):
    """
    get bounding box as token labeling (https://github.com/zihangJiang/TokenLabeling)
    return: bounding box
    """
    W = size[1] // scale
    H = size[2] // scale
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    Different with ViT use 1 conv layer, we use 4 conv layers to do patch embedding
    """

    def __init__(self, img_size=224, stem_conv=False, stem_stride=1,
                 patch_size=8, in_chans=3, hidden_dim=64, embed_dim=384):
        super().__init__()
        assert patch_size in [4, 8, 16]

        self.stem_conv = stem_conv
        if stem_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_chans, hidden_dim, kernel_size=7, stride=stem_stride,
                          padding=3, bias=False),  # 112x112
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                          padding=1, bias=False),  # 112x112
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                          padding=1, bias=False),  # 112x112
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            )

        self.proj = nn.Conv2d(hidden_dim,
                              embed_dim,
                              kernel_size=patch_size // stem_stride,
                              stride=patch_size // stem_stride)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

    def forward(self, x):
        if self.stem_conv:
            x = self.conv(x)
        x = self.proj(x)  # B, C, H, W
        return x


class Downsample(nn.Module):
    """
    Image to Patch Embedding, downsampling between stage1 and stage2
    """

    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)  # B, C, H, W
        x = x.permute(0, 2, 3, 1)
        return x


# def Intensive_blocks(block_fn, index, dim, layers):
def Intensive_blocks(block_fn, dim, layers):
    """
    generate Intensive layer in stage1
    return: Intensive layers
    """
    blocks = []
    # for block_idx in range(layers[index] // 2):
    #     # block_dpr = drop_path_rate * (block_idx +
    #     #                               sum(layers[:index])) / (sum(layers) - 1)
    #     blocks.append(block_fn(dim))
    blocks.append(block_fn(dim, layers))
    blocks = nn.Sequential(*blocks)

    return blocks


def outlooker_blocks(block_fn, index, dim, layers, num_heads=1, kernel_size=3,
                     padding=1, stride=1, mlp_ratio=3., qkv_bias=False, qk_scale=None,
                     attn_drop=0, drop_path_rate=0., **kwargs):
    """
    generate outlooker layer in stage1
    return: outlooker layers
    """
    blocks = []
    # for block_idx in range(layers[index]//2):
    for block_idx in range(layers[index]):
        # block_dpr = drop_path_rate * (block_idx +
        #                               sum(layers[:index])) / (sum(layers) - 1)
        # ZJH-modified
        block_dpr = drop_path_rate * (block_idx +
                                      sum(layers[:index])) / (sum(layers) - 1)

        blocks.append(block_fn(dim, kernel_size=kernel_size, padding=padding,
                               stride=stride, num_heads=num_heads, mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                               drop_path=block_dpr))

    blocks = nn.Sequential(*blocks)

    return blocks


def transformer_blocks(block_fn, index, dim, layers, num_heads, mlp_ratio=3.,
                       qkv_bias=False, qk_scale=None, attn_drop=0,
                       drop_path_rate=0., **kwargs):
    """
    generate transformer layers in stage2
    return: transformer layers
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx +
                                      sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(
            block_fn(dim, num_heads,
                     mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias,
                     qk_scale=qk_scale,
                     attn_drop=attn_drop,
                     drop_path=block_dpr))

    blocks = nn.Sequential(*blocks)

    return blocks


class DCAH(nn.Module):
    """
    --layers: [x,x,x,x], four blocks in two stages, the first block is outlooker, the
              other three are transformer, we set four blocks, which are easily
              applied to downstream tasks
    --img_size, --in_chans, --num_classes: these three are very easy to understand
    --patch_size: patch_size in outlook attention
    --stem_hidden_dim: hidden dim of patch embedding, d1-d4 is 64, d5 is 128
    --embed_dims, --num_heads: embedding dim, number of heads in each block
    --downsamples: flags to apply downsampling or not
    --outlook_attention: flags to apply outlook attention or not
    --mlp_ratios, --qkv_bias, --qk_scale, --drop_rate: easy to undertand
    --attn_drop_rate, --drop_path_rate, --norm_layer: easy to undertand
    --post_layers: post layers like two class attention layers using [ca, ca],
                  if yes, return_mean=False
    --return_mean: use mean of all feature tokens for classification, if yes, no class token
    --return_dense: use token labeling, details are here:
                    https://github.com/zihangJiang/TokenLabeling
    --mix_token: mixing tokens as token labeling, details are here:
                    https://github.com/zihangJiang/TokenLabeling
    --pooling_scale: pooling_scale=2 means we downsample 2x
    --out_kernel, --out_stride, --out_padding: kerner size,
                                               stride, and padding for outlook attention
    """

    def __init__(self, layers, img_size=224, in_chans=3, num_classes=1000, patch_size=8,
                 stem_hidden_dim=64, embed_dims=None, num_heads=None, downsamples=None,
                 outlook_attention=None, mlp_ratios=None, qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 post_layers=None, return_mean=False, return_dense=True, mix_token=True, assist=False,
                 pooling_scale=2, out_kernel=3, out_stride=2, out_padding=1, memory_efficient=False,
                 growth_rate=32, block_config=(6, 12, 8), num_init_features=64, bn_size=4
                 ):

        super().__init__()
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(stem_conv=True, stem_stride=2, patch_size=patch_size,
                                      in_chans=in_chans, hidden_dim=stem_hidden_dim,
                                      embed_dim=embed_dims[0])

        # inital positional encoding, we add positional encoding after outlooker blocks
        # ZJH modified: the last parameter
        self.pos_embed = nn.Parameter(
            torch.zeros(1, img_size // patch_size // pooling_scale,
                        img_size // patch_size // pooling_scale,
                        embed_dims[-1]))

        self.pos_drop = nn.Dropout(p=drop_rate)

        # set the main block in network
        network = []
        for i in range(len(layers)):
            if outlook_attention[i]:

                # ZJH
                # stage 1-1
                # stage = Intensive_blocks(IntensiveAttention, i, embed_dims[i], layers)
                # network.append(stage)

                # stage 1-2
                stage = outlooker_blocks(Outlooker, i, embed_dims[i], layers,
                                         downsample=downsamples[i], num_heads=num_heads[i],
                                         kernel_size=out_kernel, stride=out_stride,
                                         padding=out_padding, mlp_ratio=mlp_ratios[i],
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         attn_drop=attn_drop_rate, norm_layer=norm_layer)
                network.append(stage)
                # ZJH
                # stage 1-2
                # stage = Intensive_blocks(IntensiveBlock, i, embed_dims[i], layers)
                # network.append(stage)
            else:
                # stage 2
                # ZJH modified: embed_dims
                stage = transformer_blocks(Transformer, i, embed_dims[-1], layers,
                                           num_heads[i], mlp_ratio=mlp_ratios[i],
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop_path_rate=drop_path_rate,
                                           attn_drop=attn_drop_rate,
                                           norm_layer=norm_layer)
                network.append(stage)

            if downsamples[i]:
                # downsampling between two stages
                network.append(Downsample(embed_dims[i], embed_dims[i + 1], 2))

        self.network = nn.ModuleList(network)

        '''
        This part is used to realize the Dense block.
        
        '''
        den_norm_layer = nn.BatchNorm2d

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_chans, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', den_norm_layer(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            # ('relu0', nn.LeakyReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.first_AttentionCompress = nn.Conv2d(embed_dims[1], embed_dims[1] // 4, kernel_size=1, bias=False)
        # self.first_conv = TransLayer(in_chans, embed_dims[1]//4, conv_kernel_size=7, conv_stride=2, con_padding=3)
        # self.first_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.first_denseblock = IntensiveBlock(embed_dims[1]//4)
        # self.first_translayer = TransLayer(2 * embed_dims[1]//4)

        self.second_AttentionCompress = nn.Conv2d(embed_dims[1], 2 * embed_dims[1] // 4, kernel_size=1, bias=False)
        # self.second_denseblock = IntensiveBlock(2 * embed_dims[1]//4)
        # self.second_translayer = TransLayer(4 * embed_dims[1]//4)
        self.globalpooling = nn.AdaptiveAvgPool2d((1, 1))
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.attention_norm = nn.Sigmoid()

        # set post block, for example, class attention layers
        self.post_network = None
        if post_layers is not None:
            self.post_network = nn.ModuleList([
                get_block(post_layers[i],
                          dim=embed_dims[-1],
                          num_heads=num_heads[-1],
                          mlp_ratio=mlp_ratios[-1],
                          qkv_bias=qkv_bias,
                          qk_scale=qk_scale,
                          attn_drop=attn_drop_rate,
                          drop_path=0.,
                          norm_layer=norm_layer)
                for i in range(len(post_layers))
            ])
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[-1]))
            trunc_normal_(self.cls_token, std=.02)

        # set output type
        self.return_mean = return_mean  # if yes, return mean, not use class token
        self.return_dense = return_dense  # if yes, return class token and all feature tokens
        if return_dense:
            assert not return_mean, "cannot return both mean and dense"
        self.mix_token = mix_token
        self.pooling_scale = pooling_scale
        if mix_token:  # enable token mixing, see token labeling for details.
            self.beta = 1.0
            assert return_dense, "return all tokens if mix_token is enabled"
        if return_dense:
            self.aux_head = nn.Linear(
                embed_dims[-1],
                num_classes) if num_classes > 0 else nn.Identity()
        self.norm = norm_layer(embed_dims[-1])
        self.norm_128 = norm_layer(128)
        self.norm_256 = norm_layer(embed_dims[0])

        # Classifier head
        # ZJH modified: embed_dims
        self.head = nn.Linear(
            embed_dims[-1] + embed_dims[1], num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

        # self.AuxOpen = assist
        # self.AssistBlock = AssistantLayer(embed_dims[-1])
        # self.AssistBlock_2 = Assistant_Layer_After_Trans(embed_dims[-1])

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('sigmoid'))
            if m.bias is not None:
                nn.init.constant_(m.bias, val=0.0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, val=1.0)
            nn.init.constant_(m.bias, val=0.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        # patch embedding
        x = self.patch_embed(x)
        # B,C,H,W-> B,H,W,C
        x = x.permute(0, 2, 3, 1)
        return x

    def forward_tokens(self, x, ori_x):
        for idx, block in enumerate(self.network):
            # if idx == 1:
            #     feature = self.forward_assist(x)
            # ZJH If intensive block is open, idx == 3, else idx == 2
            if idx == 2:  # add positional encoding after outlooker blocks
                # if self.AuxOpen:
                #     x_aist = self.forward_assist(x)
                # else:
                #     x_aist = np.zeros([x.shape[0], self.num_classes])
                # feature, dense_feature = self.forward_assist(ori_x, x)
                feature = self.forward_assist(ori_x, x)
                # x = torch.cat((dense_feature, x), dim=3)
                x = x + self.pos_embed
                x = self.pos_drop(x)
                # feature = self.forward_assist(x)

            # if idx == 5:  # add positional
            x = block(x)
        # Add additional auxiliary block after last transformer
        # if self.AuxOpen:
        #     x_aist_2 = self.forward_assist_2(x)
        # else:
        #     x_aist_2 = np.zeros([x.shape[0], self.num_classes])
        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)
        # return x, x_aist, x_aist_2
        # return x, x_aist
        return x, feature

    def forward_assist(self, x, attention):
        attention = attention.permute(0, 3, 1, 2)
        # first-stage
        # x = self.first_conv(x)
        # x = self.first_pool(x)
        x = self.features[0:4](x)

        x = self.features[4](x)
        x = self.features[5](x)

        # Second-stage
        attention_128 = self.first_AttentionCompress(self.Upsample(attention))
        attention_128 = self.norm_128(attention_128.permute(0, 2, 3, 1))
        x = x * self.attention_norm(attention_128.permute(0, 3, 1, 2))
        x = self.features[6](x)
        x = self.features[7](x)

        # x = x.permute(0, 2, 3, 1)
        # x = self.norm_128(x)
        # x = x.permute(0, 3, 1, 2)
        # x = self.first_denseblock(x)
        # x = self.first_translayer(x)

        # Third-stage
        attention_256 = self.second_AttentionCompress(attention)
        attention_256 = self.norm_256(attention_256.permute(0, 2, 3, 1))
        # x = x + attention_256.permute(0, 3, 1, 2)
        x = x * self.attention_norm(attention_256.permute(0, 3, 1, 2))
        # dense_feature = self.norm_256(x.permute(0, 2, 3, 1))
        x = self.features[8](x)
        # x = x.permute(0, 2, 3, 1)
        # x = self.norm_256(x)
        # x = x.permute(0, 3, 1, 2)
        # x = self.second_denseblock(x)
        # x = self.second_translayer(x)

        x = self.globalpooling(x)

        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        x = x.reshape(B, C)
        #     B, H, W, C = x.shape
        #     x = x.reshape(B, -1, C)
        #     x = self.AssistBlock(x)
        # return x, dense_feature
        return x

    # def forward_assist_2(self, x):
    #     B, H, W, C = x.shape
    #     x = x.reshape(B, -1, C)
    #     x = self.AssistBlock_2(x)
    #     return x

    def forward_cls(self, x):
        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for block in self.post_network:
            x = block(x)
        return x

    def forward(self, x):
        # step1: patch embedding
        ori_x = x
        x = self.forward_embeddings(x)

        # mix token, see token labeling for details.
        if self.mix_token and self.training:
            lam = np.random.beta(self.beta, self.beta)
            patch_h, patch_w = x.shape[1] // self.pooling_scale, x.shape[
                2] // self.pooling_scale
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam, scale=self.pooling_scale)
            temp_x = x.clone()
            sbbx1, sbby1, sbbx2, sbby2 = self.pooling_scale * bbx1, self.pooling_scale * bby1, \
                                         self.pooling_scale * bbx2, self.pooling_scale * bby2
            temp_x[:, sbbx1:sbbx2, sbby1:sbby2, :] = x.flip(0)[:, sbbx1:sbbx2, sbby1:sbby2, :]
            x = temp_x
        else:
            bbx1, bby1, bbx2, bby2 = 0, 0, 0, 0

        # step2: tokens learning in the two stages
        # x, x_auxcls, x_auxcls_2 = self.forward_tokens(x)
        # x, x_auxcls = self.forward_tokens(x)
        x, feature = self.forward_tokens(x, ori_x)
        # get auxiliary classes
        # NO-4
        # x_auxcls = x_auxcls.max(1)[0]
        # NO-5
        # x_auxcls = x_auxcls.mean(1)
        # No-6
        # x_auxcls_2 = x_auxcls_2.mean(1)

        # step3: post network, apply class attention or not
        if self.post_network is not None:
            x = self.forward_cls(x)
        # else:
        #     x = self.forward_cls_without_pos(x)
        x_last_feature = x[:, 0]
        feature = self.norm(feature)
        x = self.norm(x)

        if self.return_mean:  # if no class token, return mean
            return self.head(x.mean(1))

        x_concat = torch.cat((x[:, 0], feature), dim=1)
        x_cls = self.head(x_concat)
        if not self.return_dense:
            return x_cls

        x_aux = self.aux_head(
            x[:, 1:]
        )  # generate classes in all feature tokens, see token labeling

        if not self.training:
            # return x_cls + 0.5 * x_aux.max(1)[0], x_last_feature
            # return x_cls + 0.5 * x_aux.max(1)[0]
            # ZJH xiugai
            # return x_cls + 0.25 * x_aux.max(1)[0]
            return x_cls
            # return x_auxcls

        if self.mix_token and self.training:  # reverse "mix token", see token labeling for details.
            x_aux = x_aux.reshape(x_aux.shape[0], patch_h, patch_w, x_aux.shape[-1])

            temp_x = x_aux.clone()
            temp_x[:, bbx1:bbx2, bby1:bby2, :] = x_aux.flip(0)[:, bbx1:bbx2, bby1:bby2, :]
            x_aux = temp_x

            x_aux = x_aux.reshape(x_aux.shape[0], patch_h * patch_w, x_aux.shape[-1])

        # return these: 1. class token, 2. Auxiliary class token, 4. classes from all feature tokens, 5. bounding box
        # return x_cls, x_auxcls, x_auxcls_2, x_aux, (bbx1, bby1, bbx2, bby2)
        # return x_cls, x_auxcls, x_aux, (bbx1, bby1, bbx2, bby2)
        return x_cls, x_aux, (bbx1, bby1, bbx2, bby2), x_last_feature


def dcah_net(pretrained=False, **kwargs):
    """
    DCAH-Net, Params: 86M
    """
    layers = [8, 8, 16, 4]
    # embed_dims = [128, 256, 256, 256]
    embed_dims = [256, 512, 512, 512]
    num_heads = [8, 16, 16, 16]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, False, False, False]
    outlook_attention = [True, False, False, False]
    model = DCAH(layers,
                 embed_dims=embed_dims,
                 num_heads=num_heads,
                 mlp_ratios=mlp_ratios,
                 downsamples=downsamples,
                 outlook_attention=outlook_attention,
                 post_layers=['ca', 'ca'],
                 **kwargs)
    model.default_cfg = default_cfgs['volo']
    return model
