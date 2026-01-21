# positional embedding functions

import torch


def get_precompute_cis(
        d_h: int,
        max_seq_len: int,
        device: torch.device,
        base=10000.0
    ) -> torch.Tensor:
    """
    generate complex RoPE frequency template ("cis" = "cos + i * sin")

    Arguments:
        d_h: dimension of q,k in one head
        max_seq_len: max length of sequence
    """
    theta = 1.0 / (base ** (torch.arange(0, d_h, 2, device=device) / d_h))  # [d_qk/2]
    m = torch.arange(max_seq_len, device=device)
    freqs = torch.outer(m, theta)                            # [max_seq_len, d_qk/2]   eg: [128, 32]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)   # complex number, cis = cos + isin
    return freqs_cis                                         # [max_seq_len, d_qk/2]   eg: [128, 32]


def RoPE(qk: torch.Tensor, precompute_cis: torch.Tensor,) -> torch.Tensor:
    """
    知乎版本 https://zhuanlan.zhihu.com/p/684666015

    Arguments:
        qk:             Q,K vector
        precompute_cis: complex frequency template

    Returns:
        rotated q, k
    """
    seq, d_h = qk.shape[-2], qk.shape[-1]
    qk_ = torch.view_as_complex(qk.reshape(*qk.shape[:-1], -1, 2))   # 变成复数后最后一个维度减半了，因为相邻两个元素组成了一个复数
    freqs_cis = precompute_cis[:seq, : int(d_h / 2)].reshape(1, 1, *qk_.shape[2:])  # first 2 dim: batch_size, head
    rotated_qk = qk_ * freqs_cis.cuda()
    return torch.view_as_real(rotated_qk).flatten(-2)                # 还原形状，把复数表示全部还原成了实数


def get_sinusoidal_pe_template(
        d_model: int,
        max_seq_len: int,
        device: torch.device,
        base=10000.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sinusoidal Positional Encoding precomputed template from *Attention is All You Need*.
    
    returns: `(div_term, token_position)`
    - div term `(d_model/2,)`: `1/[base^(0/d_model)], 1/[base^(2/d_model)], ..., 1/[base^(d_model/d_model)]`
    - token position range `(max_seq_len,)`: `0, 1, ..., max_seq_len-1`
    """
    freq = torch.arange(0, d_model, 2) / d_model
    div_term = torch.empty((d_model,))
    div_term = 1 / torch.pow(base, freq)
    token_position = torch.arange(max_seq_len)
    return div_term.to(device=device), token_position.to(device=device)


def get_sinusoidal_pe(
        seq_len: int,
        d_model: int,
        token_position: torch.Tensor,
        div_term: torch.Tensor,
    ) -> torch.Tensor:
    """
    compute sinusoidal positional encoding according to `seq_len`.
    
    return shape: `(seq_len, d_model)`
    """
    pe = torch.empty((seq_len, d_model), device=token_position.device)
    outer_product = torch.outer(token_position[:seq_len], div_term)  # (seq_len, d_model/2)
    pe[:, 0::2] = torch.sin(outer_product)
    pe[:, 1::2] = torch.cos(outer_product)
    return pe