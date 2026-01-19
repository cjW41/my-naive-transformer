from .modules.ffn import DeepSeekMoE
from .modules.attention import MLA
from .modules.positional_encoding import get_precompute_cis

import torch
import torch.nn.functional as F
from torch import nn


class DeepSeekV2(nn.Module):
    """
    DeepSeek-V2
    https://arxiv.org/abs/2405.04434
    """
    def __init__(self,
                 layers: int,
                 dropout: float,
                 vocab_size: int,
                 d_model: int,
                 max_seq_len: int,
                 device: torch.device,
                 n_h: int,
                 d_h:int,
                 d_c_q: int,
                 d_c_kv: int,
                 d_h_R: int,
                 d_hidden_expert: int,
                 n_shared_expert: int,
                 n_routed_expert: int,
                 n_routed_expert_activate: int,):
        """
        **Model**
        - Naive Replication of DeepSeek V2

        **Key Words**
        - DeepSeekMoE, MLA

        **Parameters**
        - Model Params: `layers`, `dropout`, `vocab_size`, `d_model`, `max_seq_len`, `device`
        - Attention Params: `n_h`, `d_c_q`, `d_c_kv`, `d_h_R`
        - MoE Params: `d_hidden_expert`,`n_shared_expert`, `n_routed_expert`, `n_routed_expert_activate`

        """
        super().__init__()
        self.layers = layers
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.device=device
        self.n_h = n_h
        self.d_h = d_h
        self.d_c_q = d_c_q
        self.d_c_kv = d_c_kv
        self.d_h_R = d_h_R
        self.d_hidden_expert = d_hidden_expert
        self.n_shared_expert = n_shared_expert
        self.n_routed_expert = n_routed_expert
        self.n_routed_expert_activate = n_routed_expert_activate

        precompute_cis = get_precompute_cis(d_h=d_h, max_seq_len=max_seq_len, device=device)

        self.attention_blocks: list[MLA] = []
        self.ffn_blocks: list[DeepSeekMoE] = []
        for _ in range(self.layers):
            self.attention_blocks.append(
                MLA(d_model=d_model,
                    n_h=n_h, d_h=d_h,
                    d_c_q=d_c_q, d_c_kv=d_c_kv,
                    d_h_R=d_h_R,
                    max_seq_len=max_seq_len,
                    precompute_cis=precompute_cis,
                    dropout=dropout, device=device,)
            )
            self.ffn_blocks.append(
                DeepSeekMoE(d_input=d_model,
                            d_output=d_model,
                            d_hidden_expert=d_hidden_expert,
                            n_shared_expert=n_shared_expert,
                            n_routed_expert=n_routed_expert,
                            n_routed_expert_activate=n_routed_expert_activate,
                            dropout=dropout, device=device,)
            )
        self.embedding = nn.Linear(vocab_size, d_model, bias=False, device=device)
        self.final_norm = nn.RMSNorm(normalized_shape=d_model, device=device)
        self.linear_outout = nn.Linear(d_model, vocab_size, bias=False)

    @classmethod
    def init_default(cls, layers: int, device: torch.device, **kwargs):
        """
        **DeepSeek-V2**

        Initialize hyperparameters according to DeepSeek-V2's technical report.
        Except for `layers`&`device`, customized parameters can be passed with kwargs.

        **Defaults**
            dropout = 0.
            vocab_size = 102400
            d_model = 5120
            max_seq_len = 4096
            n_h = 128
            d_h=128
            d_c_q = 1536
            d_c_kv = 512
            d_h_R = 64
            d_hidden_expert = 1536
            n_shared_expert = 2
            n_routed_expert = 160
            n_routed_expert_activate = 6
        """
        params = {
            "layers": layers,
            "device": device,
            "dropout": 0.,
            "vocab_size": 102400,
            "d_model": 5120,
            "max_seq_len": 4096,
            "n_h": 128,
            "d_h": 128,
            "d_c_q": 1536,
            "d_c_kv": 512,
            "d_h_R": 64,
            "d_hidden_expert": 1536,
            "n_shared_expert": 2,
            "n_routed_expert": 160,
            "n_routed_expert_activate": 6
        }
        params.update(kwargs)
        return cls(**params)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        boolean mask rule: `True -> mask`

        Arguments:
            inputs: tokenized sequence `(batch, seq, vocab_size)`
            mask: boolean padding mask `(batch, 1, 1, seq)`
        
        Returns:
            discrete distribution of next token `(batch, seq, vocab_size)`
        """
        x = self.embedding(inputs)
        for mla, moe in zip(self.attention_blocks, self.ffn_blocks):
            h = mla(x, mask)
            x = moe(h)
        return self.linear_outout(self.final_norm(x))

    @property
    def num_params(self) -> int:
        """number of parameters in DeepSeek-V2 model"""
        num_embedding = self.vocab_size * self.d_model
        num_prob_casting = self.d_model * self.vocab_size
        att, ffn = self.attention_blocks[0], self.ffn_blocks[0]
        return num_embedding + self.layers * (att.num_params + ffn.num_params) + num_prob_casting