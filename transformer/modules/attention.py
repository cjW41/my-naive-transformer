# attention modules
from transformer.modules.positional_encoding import RoPE, get_precompute_cis

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from math import sqrt


def self_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    d_scale: int,
    mask: Tensor | None,
    dropout: float,
) -> Tensor:
    """
    self attention

    Arguments:
        q,k,v: QKV tensor of shape (batch, head, seq, d_qk/d_v)
        mask: boolean mask tensor of shape `(seq, seq)`. mask rule: `True -> mask`
        d_scale: dimension of K to scale attention score
        dropout: dropout ratio
    """
    score = (q @ k.transpose(-1, -2)) / sqrt(d_scale)  # (batch, head, seq, seq)
    if mask is not None:
        score.masked_fill_(mask=mask, value=float("-inf"))
    score = F.softmax(score, dim=-1,)
    score = F.dropout(score, p=dropout,) if dropout >= 0 else score
    return score @ v  # (batch, head, seq, d_v)


def multi_head_transform(inputs: torch.Tensor, batch_size: int, n_h: int, d_h: int) -> torch.Tensor:
    """
    transform QKV to fit in the multi-head attention module

    - inputs shape:
    `(batch, seq, d_model)`
    - shape: transforming:
    `(batch, seq, d_model) -> (batch, seq, n_h, d_h)  -> (batch, n_h, seq, d_h)`

    Arguments:
        inputs: input tensor
        batch_size: batch size of `inputs`
        n_h: number of heads
        d_h: dim of one head (`n_h*d_h = d_model`)
    
    Returns:
        transformed tensor
    """
    return inputs.view(batch_size, -1, n_h, d_h).transpose(-2, -3)


def merge_heads(attention_output: torch.Tensor) -> torch.Tensor:
    """
    merge heads of attention output.

    `(batch, n_h, seq, d_h) -> (batch, seq, n_h, d_h) -> (batch, seq, n_h*d_h)`
    """
    batch_size, seq_len = attention_output.shape[0], attention_output.shape[2]
    return attention_output.transpose(1, 2).view(batch_size, seq_len, -1)


class MultiHeadAttention(nn.Module):
    """classic multi-head attention from *Attention is All You Need*"""
    def __init__(self,
                 head: int,
                 d_model: int,
                 dropout: float,
                 device: torch.device,):
        assert d_model % head == 0
        super().__init__()
        self.head = head
        self.d_model = d_model
        self.d_h = d_model // head
        self.dropout = dropout

        self.Wq = nn.Linear(d_model, d_model, device=device)
        self.Wk = nn.Linear(d_model, d_model, device=device)
        self.Wv = nn.Linear(d_model, d_model, device=device)
        self.norm = nn.LayerNorm(normalized_shape=self.d_model, device=device)

    def _multi_head_transform(self, inputs, d_h):
        """proprietary version of multi_head_transform"""
        return multi_head_transform(inputs, inputs.shape[0], n_h=self.head, d_h=d_h)

    def forward(self, q_input: Tensor, kv_input: Tensor, mask: Tensor | None) -> Tensor:
        """
        forward propogation of multihead attention
        
        Arguments:
            q_input: input tensor for Q `(batch, seq, d_model)`
            kv_input: input tensor for K,V `(batch, seq, d_model)`
            mask: combination of padding mask and causal mask, boolean tensor (`False -> mask`)
        """
        Q = self._multi_head_transform(self.Wq(q_input), d_h=self.d_h,)
        K = self._multi_head_transform(self.Wk(kv_input), d_h=self.d_h,)
        V = self._multi_head_transform(self.Wv(kv_input), d_h=self.d_h,)
        output = self_attention(Q, K, V, d_scale=self.d_h, mask=mask, dropout=self.dropout)
        output = merge_heads(output)
        return self.norm(output + q_input)  # add & norm

    @property
    def num_params(self) -> int:
        """number of parameters of multi-head attention"""
        return 3 * self.d_model * self.d_model


class MLA(nn.Module):
    """Multi-Head Latent Attention of DeepSeek-V2"""
    def __init__(self,
                 d_model: int,
                 n_h: int,
                 d_h: int,
                 d_c_q: int,
                 d_c_kv: int,
                 d_h_R: int,
                 max_seq_len: int,
                 dropout: float,
                 device: torch.device,
                 precompute_cis: torch.Tensor):
        """
        Arguments:
            d_model: the global model size
            n_h: num of heads
            d_h: dim of attention head
            d_c_q: compression dim of Q
            d_c_kv: compression dim of KV
            d_h_R: RoPE in/output vector dim for one head
            max_seq_len: RoPE parameter
        
        Notice:
            `d_model` inequal to `n_h*d_h`
        """
        assert d_model % n_h == 0
        self.d_model = d_model
        self.n_h = n_h
        self.d_h = d_h
        self.d_c_q = d_c_q
        self.d_c_kv = d_c_kv
        self.d_h_R = d_h_R
        self.d_scale = d_h + d_h_R
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.device = device
        self.precompute_cis = precompute_cis
        self.norm = nn.RMSNorm(d_model)
        # q
        self.lin_c_q = nn.Linear(d_model, d_c_q, bias=False, device=device)
        self.lin_q = nn.Linear(d_c_q, d_h * n_h, bias=False, device=device)
        self.lin_q_R = nn.Linear(d_c_q, d_h_R * n_h, bias=False, device=device)
        # kv
        self.lin_c_kv = nn.Linear(d_model, d_c_kv, bias=False, device=device)
        self.lin_k = nn.Linear(d_c_kv, d_h * n_h, bias=False, device=device)
        self.lin_k_R = nn.Linear(d_model, d_h_R, bias=False, device=device)
        self.lin_v = nn.Linear(d_c_kv, d_model, bias=False, device=device)
        # output
        self.lin_output = nn.Linear(d_h * n_h, d_model, bias=False, device=device)

    def _multi_head_transform(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        proprietary version of multi_head_transform.
        dimension of qkv heads are all `d_h`
        """
        return multi_head_transform(inputs, inputs.shape[0], n_h=self.n_h, d_h=self.d_h)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3
        seq_len = x.shape[1]
        h = self.norm(x)  # (batch, seq, d_model)
        
        # Q
        c_q = self.lin_c_q(h)
        q_c = self._multi_head_transform(self.lin_q(c_q))
        q_R = RoPE(self._multi_head_transform(self.lin_q_R(c_q)), self.precompute_cis, seq_len)
        Q = torch.stack((q_c, q_R), dim=-1)  # (batch, n_h, seq, d_h + d_h_R)
        # KV
        # Notice:
        # - all Ks use one common position embeding, a head dim is required for broadcasting
        # - shape: (batch, seq, d_model) -> (batch, seq, d_h_R) -> (batch, 1, seq, d_h_R)
        c_kv = self.lin_c_kv(h)
        k_c = self._multi_head_transform(self.lin_k(c_kv))
        k_R = RoPE(self.lin_k_R(h), self.precompute_cis, seq_len).unsqueeze(1)
        K = torch.stack((k_c, k_R), dim=-1)  # (batch, n_h, seq, d_h + d_h_R)
        V = self._multi_head_transform(self.lin_v(c_kv))

        output = self_attention(Q, K, V, d_scale=self.d_scale, mask=mask, dropout=self.dropout,)
        output = self.lin_output(merge_heads(output))  # (..., d_h*n_h) -> (..., d_model)
        return x + output

    @property
    def num_params(self) -> int:
        """number of parameters in DeepSeek-V2 MLA"""
        return (self.d_model * self.d_c_q
                + self.d_c_q * self.d_h * self.n_h
                + self.d_c_q * self.d_h_R * self.n_h
                + self.d_model * self.d_c_kv
                + self.d_c_kv * self.d_h * self.n_h
                + self.d_model * self.d_h_R
                + self.d_c_kv * self.d_model
                + self.d_h * self.n_h * self.d_model)