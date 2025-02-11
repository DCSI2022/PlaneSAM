# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            bias=True,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        return x


# # 可学习的tokens[B, 4096, 32]
# class MFA(nn.Module):
#
#     def __init__(self,
#                  theta,
#                  patch_embed_dim,
#                  num_patches,
#                  num_heads,
#                  qk_scale=None,
#                  token_dim=32
#                  ):
#         super().__init__()
#         self.num_heads = num_heads
#         self.hidden_dim = patch_embed_dim // theta  # 6
#         head_dim = self.hidden_dim // num_heads  # 2
#         self.scale = qk_scale or head_dim ** -0.5
#         self.tokens = nn.Parameter(torch.randn(1, num_patches * num_patches, token_dim), requires_grad=True)
#         self.down1 = nn.Linear(token_dim, self.hidden_dim)
#         self.down2 = nn.Linear(token_dim, self.hidden_dim)
#         self.down3 = nn.Linear(patch_embed_dim, self.hidden_dim)
#         self.LN = nn.LayerNorm(token_dim, eps=1e-6)
#         self.up = nn.Linear(self.hidden_dim, patch_embed_dim)
#
#     def forward(self, q: torch.Tensor):
#         B, N, C = q.shape
#         tokens = self.tokens.expand(B, N, 32)
#         tokens = self.LN(tokens)
#         v = self.down1(tokens).reshape(B, N, self.num_heads, self.hidden_dim // self.num_heads).permute(0, 2, 1, 3)
#         k = self.down2(tokens).reshape(B, N, self.num_heads, self.hidden_dim // self.num_heads).permute(0, 2, 1, 3)
#         q = self.down3(q).reshape(B, N, self.num_heads, self.hidden_dim // self.num_heads).permute(0, 2, 1, 3)
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         q = (attn @ v).transpose(1, 2).reshape(B, N, self.hidden_dim)
#         q = self.up(q)
#
#         return q


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias,
        qk_scale=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        # self.MFA = MFA(32, dim, 64, num_heads)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        # [B, num_heads, N, C // num_heads]
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )
        # x_r = q.permute(0, 2, 1, 3).reshape(B, N, C)
        # x_r = self.MFA(x_r)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        # x = x + x_r
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


@torch.jit.export
def get_abs_pos(
    abs_pos: torch.Tensor, has_cls_token: bool, hw: List[int]
) -> torch.Tensor:
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    h = hw[0]
    w = hw[1]
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )
        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)


class MPG(nn.Module):

    def __init__(
            self,
            beta,
            in_chans,
            kernel_size,
            use_patch_embed=True,
    ):
        super().__init__()
        self.patch_embed = None
        if use_patch_embed:
            self.patch_embed = nn.Conv2d(
                in_chans,
                in_chans,
                kernel_size=kernel_size,
                stride=1,
                padding=1,
                bias=True,
            )
        hidden_chans = int(in_chans / beta)
        self.down1 = nn.Linear(in_chans, hidden_chans)
        self.down2 = nn.Linear(in_chans, hidden_chans)
        self.up = nn.Linear(hidden_chans, in_chans)

    def forward(self, x, p, num_patches):
        """
        :param x: RGB嵌入[B, N, C]
        :param p: D嵌入[B, N, C]
        :return: [B, N, C]
        """
        if self.patch_embed:
            b, n, c = p.shape
            p = p.permute(0, 2, 1).reshape(b, c, num_patches, -1)
            p = self.patch_embed(p)
            p = p.reshape(b, c, -1).permute(0, 2, 1)

        x = self.down1(x)
        p = self.down2(p)
        p = self.up(x + p)

        return p


# Image encoder for efficient SAM.
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        patch_embed_dim: int,
        normalization_type: str,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        neck_dims: List[int],
        act_layer: Type[nn.Module],
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            patch_embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            act_layer (nn.Module): Activation layer.
        """
        super().__init__()

        self.img_size = img_size
        self.image_embedding_size = img_size // ((patch_size if patch_size > 0 else 1))
        self.transformer_output_dim = ([patch_embed_dim] + neck_dims)[-1]
        self.pretrain_use_cls_token = True
        pretrain_img_size = 224
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, patch_embed_dim)
        self.patch_embed_D = PatchEmbed(img_size, patch_size, 1, patch_embed_dim)
        # Initialize absolute positional embedding with pretrain image size.
        num_patches = (pretrain_img_size // patch_size) * (
            pretrain_img_size // patch_size
        )
        num_positions = num_patches + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, patch_embed_dim))
        self.pos_embed_D = nn.Parameter(torch.zeros(1, num_positions, patch_embed_dim))
        self.blocks = nn.ModuleList()
        for i in range(depth):
            vit_block = Block(patch_embed_dim, num_heads, mlp_ratio, True)
            self.blocks.append(vit_block)
        self.MPGs = nn.ModuleList()
        for i in range(depth):
            # 第一个MPG是没有嵌入的
            mpg_block = MPG(4, patch_embed_dim, 3, bool(i))
            self.MPGs.append(mpg_block)
        self.neck = nn.Sequential(
            nn.Conv2d(
                patch_embed_dim,
                neck_dims[0],
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(neck_dims[0]),
            nn.Conv2d(
                neck_dims[0],
                neck_dims[0],
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(neck_dims[0]),
        )

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        assert (
            x.shape[2] == self.img_size and x.shape[3] == self.img_size
        ), "input image size must match self.img_size"
        x = self.patch_embed(x)
        p = self.patch_embed_D(p)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        x = x + get_abs_pos(
            self.pos_embed, self.pretrain_use_cls_token, [x.shape[1], x.shape[2]]
        )
        p = p.permute(0, 2, 3, 1)
        p = p + get_abs_pos(
            self.pos_embed_D, self.pretrain_use_cls_token, [p.shape[1], p.shape[2]])
        num_patches = x.shape[1]
        assert x.shape[2] == num_patches
        x = x.reshape(x.shape[0], num_patches * num_patches, x.shape[3])
        p = p.reshape(p.shape[0], num_patches * num_patches, p.shape[3])
        for blk, mpg in zip(self.blocks, self.MPGs):
            p = mpg(x, p, num_patches)
            x = x + p
            x = blk(x)
        x = x.reshape(x.shape[0], num_patches, num_patches, x.shape[2])
        x = self.neck(x.permute(0, 3, 1, 2))
        return x
