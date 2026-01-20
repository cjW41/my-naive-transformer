import torch
import torch.nn.functional as F
from torch import nn, Tensor


class SwiGLU(nn.Module):
    """swish gated LU"""
    def __init__(self, d_model: int, device: torch.device):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = int(2 * d_model / 3)  # recommended hidden dim
        self.lin_gate = nn.Linear(d_model, self.d_hidden, bias=False, device=device)
        self.lin_input = nn.Linear(d_model, self.d_hidden, bias=False, device=device)
        self.lin_output = nn.Linear(self.d_hidden, d_model, bias=False, device=device)

    def forward(self, x: Tensor) -> Tensor:
        gate, gate_input = F.silu(self.lin_gate(x)), self.lin_input(x)
        return self.lin_output(gate * gate_input)

    @property
    def num_params(self) -> int:
        return 3 * self.d_model * self.d_hidden