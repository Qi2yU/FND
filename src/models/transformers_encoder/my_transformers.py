import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb
from einops import einsum, rearrange
from collections.abc import Callable, Iterable, Generator
from typing import Optional
import numpy as np
import os
import typing

class RMSNorm(nn.Module):
    def __init__(self,
                 d_model : int,
                 eps : float=1e-5,
                 device : torch.device | None=None, 
                 dtype : torch.dtype | None=None):
        super().__init__()
        self.device = device
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype))

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.weight.shape[0]
        input_dtype = x.dtype
        x = x.to(torch.float32)
        # pdb.set_trace()
        rms = ((x ** 2).sum(-1) / x.shape[-1] + self.eps).sqrt()
        # pdb.set_trace()
        rmsnorm = (x / rms.unsqueeze(-1)) * self.weight
        return rmsnorm.to(input_dtype)    


class SwinGLU_FFN(nn.Module):
    """FFN with SwiGLU, using standard nn.Linear."""

    def __init__(self, 
                 d_model: int,
                 d_ff: int | None = None,
                 device : torch.device | None=None, 
                 dtype : torch.dtype | None=None
                 ):
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            d_ff = self._cal_dff(d_model)
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        self.w1 = nn.Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = nn.Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = nn.Linear(d_model, d_ff, device=device, dtype=dtype)

    def _cal_dff(self, d_model):
        st = 8*d_model/3
        i = 1
        while(i < st):
            a, b = st - i, st + i
            if b % 64 == 0:
                return b
            elif a % 64 == 0:
                return a
    
    def forward(self, x : torch.Tensor):
        swi_i = self.w1(x)
        swi_o = swi_i * torch.sigmoid(swi_i)

        x_i = self.w3(x)

        input_final = swi_o * x_i

        output =  self.w2(input_final)
        
        return output


class RoPE_Embedding(nn.Module):
    def __init__(self, 
                 theta: float,
                 d_k: int,
                 max_seq_len: int,
                 device: torch.device | None=None):
        super().__init__()
        cos_tensor, sin_tensor = self._get_vector(theta, d_k, max_seq_len, device)
        self.register_buffer("cos_tensor", cos_tensor, persistent=False) # (max_seq_len, d_k)
        self.register_buffer("sin_tensor", sin_tensor, persistent=False)


    def _get_vector(self, theta, d_k, max_seq_len, device):
        x = torch.arange(0, max_seq_len)
        y = torch.arange(0, d_k//2)

        index_row, index_col = torch.meshgrid(x, y, indexing='ij')

        # pdb.set_trace()
        cos_tensor_single = torch.cos(index_row * torch.pow(theta, -2.0 * index_col/d_k))
        sin_tensor_single = torch.sin(index_row * torch.pow(theta, -2.0 * index_col/d_k))

        cos_tensor = cos_tensor_single.repeat_interleave(2, dim=1)
        sin_tensor = torch.stack([-sin_tensor_single, sin_tensor_single], dim=2).reshape(max_seq_len, -1)
        # pdb.set_trace()
        return cos_tensor.to(device), sin_tensor.to(device)

    def forward(self,
                x: torch.Tensor,
                token_positions: torch.Tensor) -> torch.Tensor:
        # pdb.set_trace()
        RoPE_map_cos = self.cos_tensor[token_positions] # (seq_len, d_k)
        RoPE_map_sin = self.sin_tensor[token_positions]
        
        x_reshaped = x.view(*x.shape[:-1], -1, 2) #交换x的最后一个维度相邻两个的值，计算rope结果
        x_swapped = x_reshaped[..., [1, 0]]
        x_permute = x_swapped.reshape(x.shape)

        res = x * RoPE_map_cos + x_permute * RoPE_map_sin

        return res

    
class MultiheadAttention(nn.Module):
    """Multi-head attention using nn.Linear + F.scaled_dot_product_attention.

    支持 self-attention 和 cross-attention：
      - self-attn: 只传 query (B, Lq, D)，key/value 置为 None
      - cross-attn: 传 query, key, value (B, L*, D)

    mask: (B, Lk)，1 有效，0 为 padding（作用在 key/value 侧）。
    """

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 use_RoPE: bool = False,
                 theta: int | None = None,
                 max_seq_len: int | None = None,
                 attn_dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = attn_dropout

        self.use_RoPE = use_RoPE
        if use_RoPE:
            assert theta is not None and max_seq_len is not None
            self.RoPE = RoPE_Embedding(theta, self.head_dim, max_seq_len)

    def _apply_rope(self, q, k):
        """q, k: (B, H, L, head_dim)"""
        L = q.shape[-2]
        token_positions = torch.arange(L, device=q.device)
        q = self.RoPE(q, token_positions)
        k = self.RoPE(k, token_positions)
        return q, k

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor | None = None,
                value: torch.Tensor | None = None,
                value_mask: torch.Tensor | None = None) -> torch.Tensor:
        """query/key/value: (B, L, D) ；value_mask: (B, Lk)，1 有效 0 padding。"""
        B, Lq, D = query.shape
        if key is None:
            key = query
        if value is None:
            value = key

        Lk = key.shape[1]

        # 线性投影
        q = self.q_proj(query)  # (B, Lq, D)
        k = self.k_proj(key)    # (B, Lk, D)
        v = self.v_proj(value)  # (B, Lk, D)

        # 拆分多头: (B, L, D) -> (B, H, L, Dh)
        H = self.num_heads
        Dh = self.head_dim
        q = q.view(B, Lq, H, Dh).transpose(1, 2)  # (B, H, Lq, Dh)
        k = k.view(B, Lk, H, Dh).transpose(1, 2)  # (B, H, Lk, Dh)
        v = v.view(B, Lk, H, Dh).transpose(1, 2)  # (B, H, Lk, Dh)

        # RoPE 仅在 self-attention 时启用
        if self.use_RoPE and query is key:
            q, k = self._apply_rope(q, k)

        # 展平 head 维，使用 torch 的 scaled_dot_product_attention
        q_flat = q.reshape(B * H, Lq, Dh)
        k_flat = k.reshape(B * H, Lk, Dh)
        v_flat = v.reshape(B * H, Lk, Dh)

        attn_mask = None
        if value_mask is not None:
            # value_mask: (B, Lk)，1 有效 0 padding -> True 表示 mask 掉
            key_padding = ~value_mask.to(torch.bool)  # (B, Lk)
            attn_mask = key_padding.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Lk)
            attn_mask = attn_mask.expand(B, H, Lq, Lk).reshape(B * H, Lq, Lk)

        attn_output = F.scaled_dot_product_attention(
            q_flat,
            k_flat,
            v_flat,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )  # (B*H, Lq, Dh)

        # 还原形状并输出投影
        attn_output = attn_output.reshape(B, H, Lq, Dh).transpose(1, 2)  # (B, Lq, D)
        attn_output = attn_output.reshape(B, Lq, D)
        out = self.out_proj(attn_output)
        return out


class TransformerBlock(nn.Module):
    """通用 Transformer Block：支持 self-attn / cross-attn & 可选 FFN。

    - self-attn: forward(x, mask=mask)
    - cross-attn: forward(x_q, x_kv, x_kv, mask=kv_mask)
    - use_ffn=False 时仅做注意力，不接 FFN。
    """

    def __init__(self, 
                 d_model: int, 
                 num_heads: int,
                 max_seq_len: int=None,
                 theta: int=0,
                 d_ff: int=None,
                 use_ffn: bool = False,
                 attn_dropout: float = 0.3,
                 ffn_dropout: float = 0.3):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        if d_ff is None:
            d_ff = 4 * d_model
        self.d_ff = d_ff
        self.use_ffn = use_ffn

        self.ln1 = RMSNorm(d_model)
        self.attn = MultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_RoPE=False,
            theta=theta,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
        )

        if use_ffn:
            self.ln2 = RMSNorm(d_model)
            self.ffn = SwinGLU_FFN(d_model, d_ff)
            self.ffn_dropout = nn.Dropout(ffn_dropout)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor | None = None,
                value: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """query/key/value: (B, L, D)，mask: (B, Lk)，1 有效 0 padding。

        - self-attn: forward(x, mask=mask)
        - cross-attn: forward(q, k, v, mask=kv_mask)
        """
        # 注意力子层 + 残差
        x_norm = self.ln1(query)
        attn_out = self.attn(x_norm, key=key, value=value, value_mask=mask)
        x = query + attn_out

        if not self.use_ffn:
            return x

        # FFN 子层 + 残差
        x_ffn_norm = self.ln2(x)
        ffn_out = self.ffn(x_ffn_norm)
        ffn_out = self.ffn_dropout(ffn_out)
        x = x + ffn_out
        return x

def adaptive_resize(tensor, target_len):
    return F.adaptive_avg_pool2d(tensor.transpose(1, 2), (target_len, tensor.size(2)))

class CoAttention_my(nn.Module):
    def __init__(self, model_dim=768, num_heads=8, dropout=0.5):
        super(CoAttention_my, self).__init__()
        self.attention1 = TransformerBlock(model_dim, num_heads, attn_dropout=dropout, ffn_dropout=dropout)
        self.attention2 = TransformerBlock(model_dim, num_heads, attn_dropout=dropout, ffn_dropout=dropout)
        self.linear_out = nn.Linear(2 * model_dim, model_dim)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x1, x2, mask1=None, mask2=None):
        attn_output1 = self.attention1(x1, x2, x2, mask2)
        attn_output2 = self.attention2(x2, x1, x1, mask1)
        
        # if mask1 is not None:
        #     pooled1 = masked_mean(attn_output1, mask1)   # (B, D)
        # else:
        #     pooled1 = attn_output1.mean(dim=1)

        # if mask2 is not None:
        #     pooled2 = masked_mean(attn_output2, mask2)   # (B, D)
        # else:
        #     pooled2 = attn_output2.mean(dim=1)    

        combined_1 = torch.cat([attn_output1.mean(dim=1), attn_output2.mean(dim=1) ], dim=-1)
        output_1 = self.dropout(self.linear_out(combined_1))
        output_1 = self.layer_norm(output_1)
        
        attn_output2_new = adaptive_resize(attn_output2, x1.size(1))

        combined_2 = torch.cat([attn_output1, attn_output2_new], dim=-1)
        
        output_2 = self.dropout(self.linear_out(combined_2))
        output_2 = self.layer_norm(output_2)
        
        # if mask1 is not None:
        #     output_2 = output_2 * mask1.unsqueeze(-1)   # (B, L1, D) * (B, L1, 1)

        return output_1, output_2
    

class mm_fusion(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.5):
        super(mm_fusion, self).__init__()
        self.cross_layer_1 = TransformerBlock(dim, num_heads, attn_dropout=dropout, ffn_dropout=dropout)
        self.cross_layer_2 = TransformerBlock(dim, num_heads, attn_dropout=dropout, ffn_dropout=dropout)
        self.co_layer = CoAttention_my(model_dim=dim, num_heads=num_heads, dropout=dropout)

    def forward(self, x1, x2, mask1=None, mask2=None):
        x1_, x2_ = x1, x2
        x1 = self.cross_layer_1(x1_, x2_, x2_, mask2)
        x2 = self.cross_layer_2(x2_, x1_, x1_, mask1)
        fusion_1,fusion_high_dim = self.co_layer(x1, x2, mask1, mask2)

        return fusion_1,fusion_high_dim