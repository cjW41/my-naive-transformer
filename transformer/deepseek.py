from .modules.ffn import DeepSeekMoE, DenseFFN
from .modules.attention import MLA
from .modules.positional_encoding import get_precompute_cis

import torch
from torch import nn
from typing import Literal

class DeepSeek(nn.Module):
    """
    `DeepSeek-V` family

    DeepSeek-V2:   https://arxiv.org/abs/2405.04434
    DeepSeek-V3:   https://arxiv.org/abs/2412.19437
    DeepSeek-V3.2: https://arxiv.org/abs/2512.02556
    """
    def __init__(self,
                 ver: Literal["V2", "V3", "V3.2"],
                 dense_layers: int,
                 moe_layers: int,
                 vocab_size: int,
                 d_model: int,
                 max_seq_len: int,
                 dropout: float,
                 device: torch.device,
                 *,
                 n_h: int,
                 d_h:int,
                 d_c_q: int,
                 d_c_kv: int,
                 d_hidden_dense: int = -1,
                 d_hidden_expert: int = -1,
                 n_shared_expert: int,
                 n_routed_expert: int,
                 n_routed_expert_activate: int,):
        """
        **Parameters**

        Model Params:
            dense_layers: number of starting dense layers
            moe_layers:   number of MoE layers after dense layers
            vocab_size:   vocabulary size
            d_model:      dim of embedding
            max_seq_len:  max sequence length for RoPE
        
        Attention Params:
            n_h:          number of heads in MLA
            d_h:          dim of the input into attention head
            d_c_q:        dim of latent vector for Q
            d_c_kv:       dim of latent vector for KV
        
        MoE Params:
            d_hidden_expert:          hidden layer dim of each expert
            n_shared_expert:          number of shared experts in each MoE module
            n_routed_expert:          number of routed experts in each MoE module
            n_routed_expert_activate: number of activated routed experts for each token
        """
        assert (dense_layers > 0 and d_hidden_dense > 0) or (moe_layers > 0 and d_hidden_expert > 0)
        if ver == "V3.2":
            raise NotImplementedError("DSA not implemented")
        
        super().__init__()
        self.ver = ver
        self.dense_layers = dense_layers
        self.moe_layers = moe_layers
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.device=device
        self.n_h = n_h
        self.d_h = d_h
        self.d_c_q = d_c_q
        self.d_c_kv = d_c_kv

        self.d_hidden_dense = d_hidden_dense
        self.d_hidden_expert = d_hidden_expert
        self.n_shared_expert = n_shared_expert
        self.n_routed_expert = n_routed_expert
        self.n_routed_expert_activate = n_routed_expert_activate

        precompute_cis = get_precompute_cis(d_h=d_h, max_seq_len=max_seq_len, device=device)
        self.attention_blocks: list[MLA] = []
        self.ffn_blocks: list[DeepSeekMoE | DenseFFN] = []
        # add layers with Dense FFN
        if self.dense_layers > 0:
            for _ in range(self.dense_layers):
                self.attention_blocks.append(
                    MLA(d_model=d_model,
                        n_h=n_h, d_h=d_h,
                        d_c_q=d_c_q, d_c_kv=d_c_kv,
                        max_seq_len=max_seq_len,
                        precompute_cis=precompute_cis,
                        dropout=dropout, device=device,)
                )
                self.ffn_blocks.append(
                    DenseFFN(d_input=d_model,
                             d_output=d_model,
                             d_hidden_size=d_hidden_dense,
                             device=device, dropout=dropout,)
                )
        # add layers with MoE
        if self.moe_layers > 0:
            for _ in range(self.moe_layers):
                self.attention_blocks.append(
                    MLA(d_model=d_model,
                        n_h=n_h, d_h=d_h,
                        d_c_q=d_c_q, d_c_kv=d_c_kv,
                        max_seq_len=max_seq_len,
                        precompute_cis=precompute_cis,
                        dropout=dropout, device=device,)
                )
                self.ffn_blocks.append(
                    DeepSeekMoE(d_model=d_model,
                                d_hidden=d_hidden_expert,
                                n_shared_expert=n_shared_expert,
                                n_routed_expert=n_routed_expert,
                                n_routed_expert_activate=n_routed_expert_activate,
                                dropout=dropout, device=device,
                                expert_bias = True if self.ver in ["V3", "V3.2"] else False,)
                )
        self.embedding = nn.Linear(vocab_size, d_model, bias=False, device=device)
        self.final_norm = nn.RMSNorm(normalized_shape=d_model, device=device)
        self.linear_output = nn.Linear(d_model, vocab_size, bias=False, device=device)
    
    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        boolean mask rule: `True -> mask`

        Arguments:
            inputs: tokenized sequence `(batch, seq, vocab_size)`
            mask:   boolean padding mask `(batch, 1, 1, seq)`
        
        Returns:
            discrete distribution of next token `(batch, seq, vocab_size)`
        """
        x = self.embedding(inputs)
        for mla, ffn in zip(self.attention_blocks, self.ffn_blocks):
            h = mla(x, mask)
            x = ffn(h)
        return self.linear_output(self.final_norm(x))

    @property
    def num_params(self) -> int:
        """number of parameters in DeepSeek-V2 model"""
        num_embedding = self.vocab_size * self.d_model
        num_prob_casting = self.d_model * self.vocab_size

        att = self.attention_blocks[0]
        count = num_embedding + self.moe_layers * att.num_params + num_prob_casting
        if self.moe_layers > 0:
            count += self.moe_layers * self.ffn_blocks[-1].num_params
        if self.dense_layers > 0:
            count += self.dense_layers * self.ffn_blocks[0].num_params
        return count

    @classmethod
    def init_default_V2(cls, layers: int, device: torch.device, **kwargs):
        """
        **DeepSeek-V2**
        
        Technical Report Version. Except for `layers`&`device`, customized parameters can be passed with kwargs.

        **Defaults**
            dense_layers = 0
            moe_layers = 60
            vocab_size = 102400
            d_model = 5120
            max_seq_len = 4096
            n_h = 128
            d_h=128
            d_c_q = 1536
            d_c_kv = 512
            d_hidden_expert = 1536
            n_shared_expert = 2
            n_routed_expert = 160
            n_routed_expert_activate = 6
        """
        params = {
            "ver": "V2",
            "dense_layers": 0,
            "moe_layers": layers,
            "vocab_size": 102400,
            "d_model": 5120,
            "max_seq_len": 4096,
            "device": device,
            "dropout": 0.,
            "n_h": 128,
            "d_h": 128,
            "d_c_q": 1536,
            "d_c_kv": 512,
            "d_hidden_expert": 1536,
            "n_shared_expert": 2,
            "n_routed_expert": 160,
            "n_routed_expert_activate": 6
        }
        params.update(kwargs)
        return cls(**params)

    @classmethod
    def init_default_v3(cls, dense_layers: int, moe_layers: int, device: torch.device, **kwargs):
        """"
        **DeepSeek-V3**
        
        Technical Report Version. Except for `layers`&`device`, customized parameters can be passed with kwargs.

        **Core Parameters**
            dense_layers = 58               (60 -> 58)
            moe_layers = 3                  (0 -> 3)
            vocab_size = 129280             (102400 -> 129280)
            d_model = 7168                  (5120 -> 7168)
            max_seq_len = 4096
            n_h = 128
            d_h=128
            d_c_q = 1536
            d_c_kv = 512
            d_hidden_dense = 18432          (new)
            d_hidden_expert = 2048          (1536 -> 2048)
            n_shared_expert = 1             (2 -> 1) 
            n_routed_expert = 256           (160 -> 256)
            n_routed_expert_activate = 8    (6 -> 8)
        """
        params = {
            "ver": "V3",
            "dense_layers": dense_layers,
            "moe_layers": moe_layers,
            "vocab_size": 129280,
            "d_model": 7168,
            "max_seq_len": 4096,
            "device": device,
            "dropout": 0.,
            "n_h": 128,
            "d_h": 128,
            "d_c_q": 1536,
            "d_c_kv": 512,
            "d_hidden_dense": 18432,
            "d_hidden_expert": 2048,
            "n_shared_expert": 1,
            "n_routed_expert": 256,
            "n_routed_expert_activate": 8
        }
        params.update(kwargs)
        return cls(**params)