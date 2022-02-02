from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.common_types import _size_3_t

def drop_path(
    x: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
    """
    Stochastic Depth per sample.
    Args:
        x (tensor): Input tensor.
        drop_prob (float): Probability to apply drop path.
        training (bool): If True, apply drop path to input. Otherwise (tesing), return input.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        """
        Args:
            drop_prob (float): Probability to apply drop path.
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (tensor): Input tensor.
        """
        return drop_path(x, self.drop_prob, self.training)

class SpatioTemporalEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        patch_embed_shape: Tuple[int, int, int],
        sep_pos_embed: bool = False,
        has_cls: bool = True,
    ) -> None:
        """
        Args:
            embed_dim (int): Embedding dimension for input sequence.
            patch_embed_shape (Tuple): The number of patches in each dimension
                (T, H, W) after patch embedding.
            sep_pos_embed (bool): If set to true, one positional encoding is used for
                spatial patches and another positional encoding is used for temporal
                sequence. Otherwise, only one positional encoding is used for all the
                patches.
            has_cls (bool): If set to true, a cls token is added in the beginning of each
                input sequence.
        """
        super().__init__()
        assert (
            len(patch_embed_shape) == 3
        ), "Patch_embed_shape should be in the form of (T, H, W)."
        self.cls_embed_on = has_cls
        self.sep_pos_embed = sep_pos_embed
        self._patch_embed_shape = patch_embed_shape
        self.num_spatial_patch = patch_embed_shape[1] * patch_embed_shape[2]
        self.num_temporal_patch = patch_embed_shape[0]

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            num_patches = self.num_spatial_patch * self.num_temporal_patch + 1
        else:
            num_patches = self.num_spatial_patch * self.num_temporal_patch

        if self.sep_pos_embed:
            self.pos_embed_rows = nn.Parameter(
                torch.zeros(1, self._patch_embed_shape[0], embed_dim)
            )
            self.pos_embed_cols = nn.Parameter(
                torch.zeros(1, self._patch_embed_shape[1], embed_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.num_temporal_patch, embed_dim)
            )
            if self.cls_embed_on:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

    @property
    def patch_embed_shape(self):
        return self._patch_embed_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.
        """
        B, N, C = x.shape
        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            rows_embed = self.pos_embed_rows.repeat(
                1, self._patch_embed_shape[1], 1
            ).repeat(
                1, self._patch_embed_shape[2], 1
            )

            cols_embed = self.pos_embed_cols.repeat_interleave(
                self.pos_embed_cols,
                self._patch_embed_shape[0],
                dim=1,
            ).repeat(
                1, self._patch_embed_shape[2], 1
            )

            temp_embed = torch.repeat_interleave(
                self.pos_embed_temporal,
                self.num_spatial_patch,
                dim=1,
            )
            pos_embed = rows_embed + cols_embed + temp_embed

            if self.cls_embed_on:
                pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
            x = x + pos_embed
        else:
            x = x + self.pos_embed

        return x

class Mlp(nn.Module):
    """
    A MLP block that contains two linear layers with a normalization layer. The MLP
    block is used in a transformer model after the attention block.
    ::
                         Linear (in_features, hidden_features)
                                           ↓
                                 Normalization (act_layer)
                                           ↓
                                Dropout (p=dropout_rate)
                                           ↓
                         Linear (hidden_features, out_features)
                                           ↓
                                Dropout (p=dropout_rate)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable = nn.GELU,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Args:
            in_features (int): Input feature dimension.
            hidden_features (Optional[int]): Hidden feature dimension. By default,
                hidden feature is set to input feature dimension.
            out_features (Optional[int]): Output feature dimension. By default, output
                features dimension is set to input feature dimension.
            act_layer (Callable): Activation layer used after the first linear layer.
            dropout_rate (float): Dropout rate after each linear layer. Dropout is not used
                by default.
        """
        super().__init__()
        self.dropout_rate = dropout_rate
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (tensor): Input tensor.
        """
        x = self.fc1(x)
        x = self.act(x)
        if self.dropout_rate > 0.0:
            x = self.dropout(x)
        x = self.fc2(x)
        if self.dropout_rate > 0.0:
            x = self.dropout(x)
        return x

# TODO: update attention_pool to be consistent with Improved Multiscale Vision Transformers
def _attention_pool(
    tensor: torch.Tensor,
    pool: Optional[Callable],
    thw_shape: List[int],
    has_cls_embed: bool = True,
    norm: Optional[Callable] = None,
) -> torch.Tensor:
    """
    Apply pool to a flattened input (given pool operation and the unflattened shape).
                                         Input
                                           ↓
                                        Reshape
                                           ↓
                                          Pool
                                           ↓
                                        Reshape
                                           ↓
                                          Norm
    Args:
        tensor (torch.Tensor): Input tensor.
        pool (Optional[Callable]): Pool operation that is applied to the input tensor.
            If pool is none, return the input tensor.
        thw_shape (List): The shape of the input tensor (before flattening).
        has_cls_embed (bool): Whether the input tensor contains cls token. Pool
            operation excludes cls token.
        norm: (Optional[Callable]): Optional normalization operation applied to tensor
            after pool.
    Returns:
        tensor (torch.Tensor): Input tensor after pool.
        thw_shape (List[int]): Output tensor shape (before flattening).
    """
    if pool is None:
        return tensor, thw_shape
    tensor_dim = tensor.ndim
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")

    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    tensor, block_shape = block_to_image(tensor, thw_shape)

    tensor = pool(tensor)

    tensor, thw_shape = image_to_block(tensor, block_shape)

    if has_cls_embed:
        tensor = torch.cat((cls_tok, tensor), dim=2)
    if norm is not None:
        tensor = norm(tensor)

    if tensor_dim == 4:
        pass
    else:  # For the case tensor_dim == 3.
        tensor = tensor.squeeze(1)

    return tensor, thw_shape

def block_to_image(x, patch_shape) -> Tuple[torch.Tensor, tuple]:
    B, N, L, C = x.shape
    T, H, W = patch_shape
    x = x.reshape(B * N, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
    return x, [B, N, L, C]

def image_to_block(x, block_shape) -> Tuple[torch.Tensor, list]:
    patch_shape = [x.shape[2], x.shape[3], x.shape[4]]
    L_pooled = np.prod(patch_shape)
    x = x.reshape(block_shape[0], block_shape[1], x.shape[1], L_pooled).transpose(2, 3)
    return x, patch_shape

class MultiScaleAttention(nn.Module):
    """
    Implementation of a multiscale attention block. Compare to a conventional attention
    block, a multiscale attention block optionally supports pooling (either
    before or after qkv projection). If pooling is not used, a multiscale attention
    block is equivalent to a conventional attention block.
    ::
                                   Input
                                     |
                    |----------------|-----------------|
                    ↓                ↓                 ↓
                  Linear           Linear            Linear
                    &                &                 &
                 Pool (Q)         Pool (K)          Pool (V)
                |---|                |                 |
                |    |                + R_{p(i), p(j)}  |
                |    → -------------- ←                 |
                |             ↓                         |
                |       MatMul & Scale                  |
                |             ↓                         |
                |          Softmax                      |
                |             → ----------------------- ←
                |                         ↓
                |                   MatMul & Scale
                |                         ↓
                |                       DropOut
                |                         ↓
                → ----------------------→ +
                                          |
                                         Norm
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        dropout_rate: float = 0.0,
        kernel_q: _size_3_t = (1, 1, 1),
        kernel_kv: _size_3_t = (1, 1, 1),
        stride_q: _size_3_t = (1, 1, 1),
        stride_kv: _size_3_t = (1, 1, 1),
        upscale_factor: int = 1,
        norm_layer: Callable = nn.LayerNorm,
        has_cls_embed: bool = True,
        pool_mode: str = "conv",
        pool_first: bool = False,
    ) -> None:
        """
        Args:
            dim (int): Input feature dimension.
            num_heads (int): Number of heads in the attention layer.
            qkv_bias (bool): If set to False, the qkv layer will not learn an additive
                bias. Default: False.
            dropout_rate (float): Dropout rate.
            kernel_q (_size_3_t): Pooling kernel size for q. If both pooling kernel
                size and pooling stride size are 1 for all the dimensions, pooling is
                disabled.
            kernel_kv (_size_3_t): Pooling kernel size for kv. If both pooling kernel
                size and pooling stride size are 1 for all the dimensions, pooling is
                disabled.
            stride_q (_size_3_t): Pooling kernel stride for q.
            stride_kv (_size_3_t): Pooling kernel stride for kv.
            norm_layer (nn.Module): Normalization layer used after pooling.
            has_cls_embed (bool): If set to True, the first token of the input tensor
                should be a cls token. Otherwise, the input tensor does not contain a
                cls token. Pooling is not applied to the cls token.
            pool_mode (str): Pooling mode. Option includes "conv" (learned pooling), "avg"
                (average pooling), and "max" (max pooling).
            pool_first (bool): If set to True, pool is applied before qkv projection.
                Otherwise, pool is applied after qkv projection. Default: False.
        """

        super().__init__()
        assert pool_mode in ["conv", "avg", "max"]

        self.pool_first = pool_first
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.has_cls_embed = has_cls_embed
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if dropout_rate > 0.0:
            self.proj_drop = nn.Dropout(dropout_rate)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if (
            kernel_q is not None
            and np.prod(kernel_q) == 1
            and np.prod(stride_q) == 1
        ):
            kernel_q = None
        if (
            kernel_kv is not None
            and np.prod(kernel_kv) == 1
            and np.prod(stride_kv) == 1
        ):
            kernel_kv = None

        if pool_mode in ("avg", "max"):
            pool_op = nn.MaxPool3d if pool_mode == "max" else nn.AvgPool3d
            self.pool_q = (
                pool_op(kernel_q, stride_q, padding_q, ceil_mode=False)
                if kernel_q is not None
                else None
            )
            self.pool_k = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if kernel_kv is not None
                else None
            )
            self.pool_v = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if kernel_kv is not None
                else None
            )
        elif pool_mode == "conv":
            self.pool_q = (
                nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=head_dim,
                    bias=False,
                )
                if kernel_q is not None
                else None
            )
            self.norm_q = norm_layer(head_dim) if kernel_q is not None else None
            self.pool_k = (
                nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=head_dim,
                    bias=False,
                )
                if kernel_kv is not None
                else None
            )
            self.norm_k = norm_layer(head_dim) if kernel_kv is not None else None
            self.pool_v = (
                nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=head_dim,
                    bias=False,
                )
                if kernel_kv is not None
                else None
            )
            self.norm_v = norm_layer(head_dim) if kernel_kv is not None else None
        else:
            raise NotImplementedError(f"Unsupported model {pool_mode}")

    def _qkv_proj(
        self,
        q: torch.Tensor,
        q_size: List[int],
        k: torch.Tensor,
        k_size: List[int],
        v: torch.Tensor,
        v_size: List[int],
        batch_size: List[int],
        chan_size: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = (
            self.q(q)
            .reshape(batch_size, q_size, self.num_heads, chan_size // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(k)
            .reshape(batch_size, k_size, self.num_heads, chan_size // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(v)
            .reshape(batch_size, v_size, self.num_heads, chan_size // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        return q, k, v

    def _qkv_pool(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        thw_shape: Tuple[torch.Tensor, List[int]],
    ) -> Tuple[
        torch.Tensor, List[int], torch.Tensor, List[int], torch.Tensor, List[int]
    ]:
        q, q_shape = _attention_pool(
            q,
            self.pool_q,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_q if hasattr(self, "norm_q") else None,
        )
        k, k_shape = _attention_pool(
            k,
            self.pool_k,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_k if hasattr(self, "norm_k") else None,
        )
        v, v_shape = _attention_pool(
            v,
            self.pool_v,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_v if hasattr(self, "norm_v") else None,
        )
        return q, q_shape, k, k_shape, v, v_shape

    def _get_qkv_length(
        self,
        q_shape: List[int],
        k_shape: List[int],
        v_shape: List[int],
    ) -> Tuple[int]:
        q_N = np.prod(q_shape) + 1 if self.has_cls_embed else np.prod(q_shape)
        k_N = np.prod(k_shape) + 1 if self.has_cls_embed else np.prod(k_shape)
        v_N = np.prod(v_shape) + 1 if self.has_cls_embed else np.prod(v_shape)
        return q_N, k_N, v_N

    def _reshape_qkv_to_seq(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_N: int,
        v_N: int,
        k_N: int,
        B: int,
        C: int,
    ) -> Tuple[int]:
        q = q.permute(0, 2, 1, 3).reshape(B, q_N, C)
        v = v.permute(0, 2, 1, 3).reshape(B, v_N, C)
        k = k.permute(0, 2, 1, 3).reshape(B, k_N, C)
        return q, k, v

    def forward(
        self, x: torch.Tensor, thw_shape: List[int]
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Args:
            x (torch.Tensor): Input tensor.
            thw_shape (List): The shape of the input tensor (before flattening).
        """

        B, N, C = x.shape
        if self.pool_first:
            x = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q = k = v = x
            q, q_shape, k, k_shape, v, v_shape = self._qkv_pool(q, k, v, thw_shape)
            q_N, k_N, v_N = self._get_qkv_length(q_shape, k_shape, v_shape)
            q, k, v = self._reshape_qkv_to_seq(q, k, v, q_N, v_N, k_N, B, C)
            q, k, v = self._qkv_proj(q, q_N, k, k_N, v, v_N, B, C)
        else:
            q = k = v = x
            q, k, v = self._qkv_proj(q, N, k, N, v, N, B, C)
            q, q_shape, k, k_shape, v, v_shape = self._qkv_pool(q, k, v, thw_shape)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        N = q.shape[2]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        if self.dropout_rate > 0.0:
            x = self.proj_drop(x)
        return x, q_shape
