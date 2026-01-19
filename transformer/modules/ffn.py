# ffn module of Transformer

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from torch.distributions import Categorical


from typing import Literal, Callable


class SwiGLU(nn.Module):
    """swish gated LU"""
    def __init__(self, d_model: int, device: torch.device):
        super().__init__()
        d_hidden = 2 * d_model
        self.lin_gate = nn.Linear(d_model, d_hidden, bias=False, device=device)
        self.lin_input = nn.Linear(d_model, d_hidden, bias=False, device=device)
        self.lin_output = nn.Linear(d_hidden, d_model, bias=False, device=device)

    def forward(self, x: Tensor) -> Tensor:
        gate, gate_input = F.silu(self.lin_gate(x)), self.lin_input(x)
        return self.lin_output(gate * gate_input)


class DenseFFN(nn.Module):
    """
    Dense FFN of Classic Transformer (input RMSNorm, res conn)
    
    default activation: SwiGLU
    """
    def __init__(self,
                 d_input: int,
                 d_output: int,
                 d_hidden_size: int,
                 device: torch.device,
                 dropout: float,
                 activation: nn.Module | None = None):
        super().__init__()        
        self.d_input = d_input
        self.d_output = d_output
        self.d_hidden_size = d_hidden_size

        self.net = nn.Sequential(
            nn.RMSNorm(normalized_shape=d_input, device=device),
            nn.Linear(d_input, d_hidden_size, device=device),
            activation or SwiGLU(d_model=d_output, device=device),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(d_hidden_size, d_output, device=device),
        )
    
    def forward(self, x):
        return  x + self.net(x)
    
    @property
    def num_params(self) -> int:
        """number of parameters in dense FFN module"""
        return (self.d_input + self.d_output) * self.d_hidden_size


class DeepSeekMoE(nn.Module):
    """MoE of DeepSeek V2"""
    def __init__(self,
                 d_input: int,
                 d_output: int,
                 d_hidden_expert: int,
                 n_shared_expert: int,
                 n_routed_expert: int,
                 n_routed_expert_activate: int,
                 dropout: float,
                 device: torch.device):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.d_hidden_expert = d_hidden_expert
        self.n_shared_expert = n_shared_expert
        self.n_routed_expert = n_routed_expert
        self.n_routed_expert_activate = n_routed_expert_activate

        self.norm = nn.RMSNorm(normalized_shape=d_input)
        # shared experts
        self.shared_experts = nn.ModuleList(
            [
                DenseFFN(d_input=d_input, d_output=d_output, d_hidden_size=d_hidden_expert, dropout=dropout, device=device)
                for _ in range(n_shared_expert)
            ]
        )

        # routed experts
        self.router = nn.Linear(d_input, n_routed_expert, bias=False, device=device)
        self.routed_experts = nn.ModuleList(
            [
                DenseFFN(d_input=d_input, d_output=d_output, d_hidden_size=d_hidden_expert, dropout=dropout, device=device)
                for _ in range(n_routed_expert)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        inputs = self.norm(x)
        
        # Calculate Shared Experts
        # sum shared experts' output with weight 1
        shared_output_list = [exp(inputs) for exp in self.shared_experts]
        shared_output = torch.sum(torch.stack(shared_output_list, dim=0), dim=0)

        # Calculate Routed Experts
        # sum routed experts' output with selection probs as weight (routed experts count as merely 1 expert in final output)
        # - selected_logits: selected experts' logits in descending order
        # - indice: selected experts' index ordered like selected_logits
        logits = self.router(inputs)  # (batch, seq, d_model) -> (batch, seq, n_routed_expert)
        selected_logits, indice = torch.topk(logits, k=self.n_routed_expert_activate, dim=-1)
        selected_output_list = [
            exp(inputs)
            for idx, exp in enumerate(self.routed_experts)
            if idx in indice
        ]
        weight = F.softmax(selected_logits, dim=-1).unsqueeze(dim=-2)  # (batch, seq, 1, n_routed_expert_activate)
        stacked_output = torch.stack(selected_output_list, dim=-2)  # (batch, seq, d_model) -> (batch, seq, n_routed_expert_activate, d_model)
        routed_output = (weight @ stacked_output).squeeze()  # (batch, seq, 1, d_model) -> (batch, seq, d_model)

        return x + shared_output + routed_output

    @property
    def num_params(self) -> int:
        """number of parameters in MoE module"""
        share, routed = self.shared_experts[0], self.routed_experts[0]
        assert isinstance(share, DenseFFN) and isinstance(routed, DenseFFN)
        return self.n_shared_expert * share.num_params + self.n_routed_expert * routed.num_params