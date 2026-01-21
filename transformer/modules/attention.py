from .positional_encoding import RoPE

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
        q,k,v:   QKV tensor of shape `(batch, n_h, seq, d_h)`
        mask:    boolean mask tensor `(seq, seq)`. boolean mask rule: `True -> mask`.
        d_scale: dimension of K to scale attention score
        dropout: dropout ratio
    """
    score = (q @ k.transpose(-1, -2)) / sqrt(d_scale)  # (batch, n_h, seq, seq)
    if mask is not None:
        score.masked_fill_(mask=mask, value=float("-inf"))
    score = F.softmax(score, dim=-1,)
    score = F.dropout(score, p=dropout,) if dropout >= 0 else score
    return score @ v  # (batch, n_h, seq, d_h)


def multi_head_transform(inputs: torch.Tensor, batch_size: int, n_h: int, d_h: int) -> torch.Tensor:
    """
    transform QKV to fit in the multi-head attention module

    shape change: `(batch, seq, d_model) -> (batch, seq, n_h, d_h)  -> (batch, n_h, seq, d_h)`

    Arguments:
        inputs:     input tensor `(batch, seq, d_model)`
        batch_size: batch size of `inputs`
        n_h:        number of heads
        d_h:        dim of one head (`n_h*d_h = d_model`)
    
    Returns:
        transformed tensor
    """
    return inputs.reshape(batch_size, -1, n_h, d_h).transpose(-2, -3)


def merge_heads(attention_output: torch.Tensor) -> torch.Tensor:
    """
    merge output of multi-head attention heads

    shape change: `(batch, n_h, seq, d_h) -> (batch, seq, n_h, d_h) -> (batch, seq, n_h*d_h)`
    """
    batch_size, seq_len = attention_output.shape[0], attention_output.shape[2]
    return attention_output.transpose(1, 2).reshape(batch_size, seq_len, -1)


class MultiHeadAttention(nn.Module):
    """classic multi-head attention with RoPE support"""
    def __init__(self,
                 n_h: int,
                 d_qk: int,
                 d_v: int,
                 d_model: int,
                 dropout: float,
                 device: torch.device,
                 use_rope: bool = False,
                 precompute_cis: torch.Tensor | None = None,):
        super().__init__()
        # Dimension of QKV:
        # qk must have same dim for dot-product. v can have a different dim.
        # it is recommended to set `d_v = d_qk = d_model/n_h`
        self.n_h = n_h
        self.d_qk = d_qk          
        self.d_v = d_v            
        self.d_model = d_model
        self.dropout = dropout
        self.use_rope = use_rope
        self.precompute_cis = precompute_cis

        self.Wq = nn.Linear(d_model, n_h * d_qk, bias=False, device=device)
        self.Wk = nn.Linear(d_model, n_h * d_qk, bias=False, device=device)
        self.Wv = nn.Linear(d_model, n_h * d_v, bias=False, device=device)
        self.Wo = nn.Linear(n_h * d_v, d_model, bias=False, device=device)  # restore shape, mix output of heads
        self.norm = nn.LayerNorm(normalized_shape=self.d_model, device=device)

    def _multi_head_transform(self, inputs, d_h):
        """proprietary version of multi_head_transform"""
        return multi_head_transform(inputs, inputs.shape[0], n_h=self.n_h, d_h=d_h)

    def forward(
        self,
        q_input: Tensor,
        kv_input: Tensor,
        mask: Tensor | None,
    ) -> Tensor:
        """
        forward propogation of multihead attention
        
        Arguments:
            q_input:  input tensor for Q `(batch, seq, d_model)`
            kv_input: input tensor for K,V `(batch, seq, d_model)`
            mask:     boolean mask for QK^T `(batch, 1, seq)` (broadcasting along 'column')
        """
        Q = self._multi_head_transform(self.Wq(q_input), d_h=self.d_qk)
        K = self._multi_head_transform(self.Wk(kv_input), d_h=self.d_qk)
        V = self._multi_head_transform(self.Wv(kv_input), d_h=self.d_v)
        if self.use_rope:
            assert self.precompute_cis is not None
            Q = RoPE(Q, self.precompute_cis,)
            K = RoPE(K, self.precompute_cis,)
        output = self_attention(Q, K, V, d_scale=self.d_v, mask=mask, dropout=self.dropout)  # (batch, seq, n_h*d_v)
        output = self.Wo(merge_heads(output))  # (..., n_h*d_v) -> (..., d_model)
        return self.norm(output + q_input)     # add & norm

    @property
    def num_params(self) -> int:
        """number of parameters of multi-head attention"""
        return 3 * self.d_model * self.d_model


class MLA(nn.Module):
    """Multi-Head Latent Attention"""
    def __init__(self,
                 d_model: int,
                 n_h: int,
                 d_h: int,
                 d_c_q: int,
                 d_c_kv: int,
                 max_seq_len: int,
                 dropout: float,
                 device: torch.device,
                 precompute_cis: torch.Tensor,):
        """
        Arguments:
            d_model:         token dim
            n_h:             num of heads
            d_h:             dim of attention head
            d_c_q:           compression dim of Q
            d_c_kv:          compression dim of KV
            max_seq_len:     RoPE parameter
        
        Partial RoPE:
            RoPE is only implemented on half of the input of attention head.
            Firstly, create two tensors with trailing dim `d_h/2`.
            Then, apply RoPE to one tensor, concat it with the other one along the trailing axis.
        """
        super().__init__()
        self.d_model = d_model
        self.n_h = n_h
        self.d_h = d_h
        self.d_c_q = d_c_q
        self.d_c_kv = d_c_kv
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.device = device
        
        self.precompute_cis = precompute_cis
        self.input_norm = nn.RMSNorm(d_model, device=device)
        # Additional Norm
        # """
        #  [DeepSeek-V2]
        #  In addition, the low-rank compression ... will impact the output scale of a layer.
        #  Therefore, in practice, we employ additional RMS Norm layers after the compressed latent vectors, ...
        #  (i.e., the compressed latent vectors ...) to ensure stable training.
        # """
        self.c_q_norm = nn.RMSNorm(d_c_q, device=device)
        self.c_kv_norm = nn.RMSNorm(d_c_kv, device=device)

        # q
        self.lin_c_q = nn.Linear(d_model, d_c_q, bias=False, device=device)
        self.lin_q = nn.Linear(d_c_q, int(d_h / 2) * n_h, bias=False, device=device)
        self.lin_q_R = nn.Linear(d_c_q, int(d_h / 2) * n_h, bias=False, device=device)

        # kv
        self.lin_c_kv = nn.Linear(d_model, d_c_kv, bias=False, device=device)
        self.lin_k = nn.Linear(d_c_kv, int(d_h / 2) * n_h, bias=False, device=device)
        self.lin_k_R = nn.Linear(d_model, int(d_h / 2), bias=False, device=device)
        self.lin_v = nn.Linear(d_c_kv, d_h * n_h, bias=False, device=device)

        # output
        self.Wo = nn.Linear(d_h * n_h, d_model, bias=False, device=device)

    def _multi_head_transform(self, inputs: torch.Tensor, d_h: int) -> torch.Tensor:
        """proprietary version of multi_head_transform."""
        return multi_head_transform(inputs, inputs.shape[0], n_h=self.n_h, d_h=d_h)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.input_norm(x)             # -> (batch, seq, d_model)
        
        # Q
        c_q = self.c_q_norm(self.lin_c_q(h))
        q_c = self._multi_head_transform(self.lin_q(c_q), d_h = int(self.d_h / 2))
        q_R = RoPE(self._multi_head_transform(self.lin_q_R(c_q), d_h = int(self.d_h / 2)), self.precompute_cis,)
        Q = torch.cat((q_c, q_R), dim=-1)  # -> (batch, n_h, seq, d_h)
        # KV
        c_kv = self.c_kv_norm(self.lin_c_kv(h))
        V = self._multi_head_transform(self.lin_v(c_kv), d_h = self.d_h)
        # Notice:
        #    all attention heads use one common position embeding, a head dim is required
        #    (batch, seq, d_model) -> (batch, seq, d_h/2) -> (batch, 1, seq, d_h/2)
        k_c = self._multi_head_transform(self.lin_k(c_kv), d_h = int(self.d_h / 2))
        k_R = RoPE(self.lin_k_R(h).unsqueeze(1), self.precompute_cis,)  # unsqueeze to add a head dim for broadcasting
        k_R = k_R.repeat(1, self.n_h, 1, 1)                             # repeat along head dim to fit in torch.stack
        K = torch.cat((k_c, k_R), dim=-1)                               # (batch, n_h, seq, d_h)

        output = self_attention(Q, K, V, d_scale=self.d_h, mask=mask, dropout=self.dropout,)
        output = self.Wo(merge_heads(output))                           # (..., d_h*n_h) -> (..., d_model)
        return x + output

    @property
    def num_params(self) -> int:
        return (self.d_model * self.d_c_q
                + self.d_c_q * self.d_h * self.n_h
                + self.d_model * self.d_c_kv
                + self.d_c_kv * int(self.d_h / 2) * self.n_h
                + self.d_model * int(self.d_h / 2)
                + self.d_c_kv * self.d_model
                + self.d_h * self.n_h * self.d_model)


class GDA(nn.Module):
    """Gated Delta Attention of Qwen-Next"""
    pass


class KDA(nn.Module):
    """Kimi Delta Attention of Kimi-Linear"""
    pass
