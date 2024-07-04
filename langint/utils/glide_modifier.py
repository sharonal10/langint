import torch.nn as nn
import torch.nn.functional as F
from glide_text2im.unet import QKVAttention
from glide_text2im.xf import QKVMultiheadAttention
from typing import List
import torch


def hook_attention(self: QKVAttention, qkv, encoder_kv):
    # https://github.com/openai/glide-text2im/blob/69b530740eb6cef69442d6180579ef5ba9ef063e/glide_text2im/unet.py#L261

    bs, width, length = qkv.shape
    assert width % (3 * self.n_heads) == 0
    ch = width // (3 * self.n_heads)
    q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)  # (N * H, C, T or S)
    if encoder_kv is not None:
        assert encoder_kv.shape[1] == self.n_heads * ch * 2
        ek, ev = encoder_kv.reshape(bs * self.n_heads, ch * 2, -1).split(ch, dim=1)
        k = torch.cat([ek, k], dim=-1)
        v = torch.cat([ev, v], dim=-1)

    q = q.permute(0, 2, 1)  # -> (N * H, T, C)
    k = k.permute(0, 2, 1)  # -> (N * H, S, C)
    v = v.permute(0, 2, 1)  # -> (N * H, S, C)
    if q.dtype != torch.float32:  # ?
        raise NotImplementedError()
        a = F.scaled_dot_product_attention(
            query=q.float(), key=k.float(), value=v.float(),
            attn_mask=torch.zeros((), dtype=torch.float32, device='cuda')
        ).to(q.dtype)
    else:
        a = F.scaled_dot_product_attention(
            query=q, key=k, value=v,
            attn_mask=torch.zeros((), dtype=q.dtype, device='cuda')
        )
    a = a.permute(0, 2, 1)  # (N * H, T, C) -> (N * H, C, T)
    a = a.reshape(bs, -1, length)
    return a


def compare_attention(self, input, output):
    a = hook_attention(self, *input)
    if not torch.allclose(a, output):
        # import ipdb; ipdb.set_trace()
        print('attn', (a - output).abs().max())


def hook_multihead_attention(self: QKVMultiheadAttention, qkv):
    bs, n_ctx, width = qkv.shape
    attn_ch = width // self.n_heads // 3
    qkv = qkv.view(bs, n_ctx, self.n_heads, -1)  # (B, T or S, H, C * 3)
    q, k, v = torch.split(qkv, attn_ch, dim=-1)  # (B, T or S, H, C)
    q = q.permute(0, 2, 1, 3)  # -> (B, H, T, C)
    k = k.permute(0, 2, 1, 3)  # -> (B, H, S, C)
    v = v.permute(0, 2, 1, 3)  # -> (B, H, S, C)
    a = F.scaled_dot_product_attention(
        query=q, key=k, value=v,
        attn_mask=torch.zeros((), dtype=q.dtype, device='cuda')
    )  # (B, H, T, C)
    a = a.permute(0, 2, 1, 3)  # (B, T, H, C)
    a = a.reshape(bs, n_ctx, -1)
    return a


def compare_multihead_attention(self, input, output):
    a = hook_multihead_attention(self, *input)
    if not torch.allclose(a, output):
        # import ipdb; ipdb.set_trace()
        print('multattn', (a - output).abs().max())


@torch.no_grad()
def change_attention_debug(m):
    if isinstance(m, QKVAttention):
        m.register_forward_hook(compare_attention)
    elif isinstance(m, QKVMultiheadAttention):
        m.register_forward_hook(compare_multihead_attention)
    else:
        pass


class AttnHook:
    # modify forward function
    def __init__(self, m: QKVAttention):
        self.module = m
        m.forward = self.forward

    def forward(self, qkv, encoder_kv=None):
        return hook_attention(self.module, qkv, encoder_kv)


class MultiAttnHook:
    def __init__(self, m: QKVMultiheadAttention):
        self.module = m

    def forward(self, qkv):
        return hook_multihead_attention(self.module, qkv)


def change_attention(m):
    if isinstance(m, QKVAttention):
        AttnHook(m)
    elif isinstance(m, QKVMultiheadAttention):
        MultiAttnHook(m)
    else:
        pass