from .activation import SwiGLU

import torch
import torch.nn.functional as F
from torch import nn
from math import sqrt


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
        self.d_input = d_input    # normally d_input = d_output
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
        return (self.d_input + self.d_output) * self.d_hidden_size


class BatchedExpertsBase(nn.Module):
    """
    Batch-Processing Expert Group for MoE
    Base class of `SharedExperts` & `RoutedExperts`
    """
    def __init__(self,
                 n_expert: int,
                 d_input: int,
                 d_output: int,
                 d_hidden: int,
                 dropout: float,
                 device: torch.device,):
        """
        Arguments:
            n_expert:   total number of experts
            d_input:    FFN input dim
            d_output:   FFN output dim
            d_hidden:   FFN hidden dim
        """
        super().__init__()
        self.n_expert = n_expert
        self.d_input = d_input
        self.d_output = d_output
        self.d_hidden = d_hidden
        self.dropout = dropout
        
        # weight&bias of FFN layer with batched dim: n_expert
        W1 = nn.Parameter(
            torch.empty((n_expert, d_input, d_hidden), device=device),
            requires_grad=True,
        )
        b1 = nn.Parameter(
            torch.zeros((n_expert, 1, d_hidden), device=device),  # broadcast along the 2nd dim
            requires_grad=True,
        )
        W2 = nn.Parameter(
            torch.empty((n_expert, d_hidden, d_output), device=device),
            requires_grad=True,
        )
        b2 = nn.Parameter(
            torch.zeros((n_expert, 1, d_output), device=device),
            requires_grad=True,
        )

        # initialize weights
        bound1, bound2 = 1/sqrt(d_input), 1/sqrt(d_output)
        nn.init.uniform_(W1, -bound1, bound1)
        nn.init.uniform_(W2, -bound2, bound2)

        # model
        self.W1, self.b1 = W1, b1
        self.act1 = SwiGLU(d_model=d_hidden, device=device)
        self.W2, self.b2 = W2, b2

    def forward(self, x):
        raise NotImplementedError("use SharedExperts, RoutedExperts instead")

    @property
    def num_params(self) -> int:
        return (self.n_expert * self.d_input * self.d_hidden
                + self.n_expert * self.d_hidden
                + self.n_expert * self.d_hidden * self.d_output
                + self.n_expert * self.d_output) + self.act1.num_params


class SharedExperts(BatchedExpertsBase):
    
    def __init__(self,
                 n_expert: int,
                 d_input: int,
                 d_output: int,
                 d_hidden: int,
                 dropout: float,
                 device: torch.device,):
        super().__init__(
            n_expert=n_expert,
            d_input=d_input,
            d_output=d_output,
            d_hidden=d_hidden,
            dropout=dropout,
            device=device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: MLA output (batch, seq, d_input)
        Returns:
            batched experts' output in shape (n_expert, batch, seq, d_output)
        """
        # Step1: transform x to fit in batched dim n_expert

        # (batch, seq, d_input) -> (batch*seq, d_input) -> (1, batch*seq, d_input)
        batch, seq, d_input = x.shape
        x = x.reshape(-1, d_input).unsqueeze(0)
        
        # Step2: FFN

        # additional scaling
        # """
        #  [DeepSeek-V2]
        #  In addition, ... and fine-grained expert segmentation will impact the output scale of a layer.
        #  Therefore, in practice, we ... and multiply additional scaling factors at the width bottlenecks
        #   (i.e., ... the intermediate hidden states of routed experts) to ensure stable training.
        # """
        # The explicit method of scaling remains unknown. It shall be ignored.
        h = self.act1(x @ self.W1 + self.b1)  # -> (n_expert, batch*seq, d_hidden)
        if self.dropout > 0:
            h = F.dropout(h, p=self.dropout)
        output = h @ self.W2 + self.b2        # -> (n_expert, batch*seq, d_output)
        return output.reshape(self.n_expert, batch, seq, self.d_output)


class RoutedExperts(BatchedExpertsBase):
    
    def __init__(self,
                 n_expert: int,
                 n_activate: int,
                 d_input: int,
                 d_output: int,
                 d_hidden: int,
                 dropout: float,
                 device: torch.device,):
        super().__init__(
            n_expert=n_expert,
            d_input=d_input,
            d_output=d_output,
            d_hidden=d_hidden,
            dropout=dropout,
            device=device
        )
        self.n_activate = n_activate

    def forward(self, x: torch.Tensor, affinity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            x:        MLA output `(batch, seq, d_input)`.
            affinity: normalized token-to-expert affinity generated by MoE router (batch, seq, n_expert)
        Return:
            output:            batched experts' output in shape (batch, seq, n_activate, d_output)
            activated_indices: top-k experts' indices **for each token in each batch** (batch, seq, n_activate)
        """
        # Step1: Input Transformation

        # batch-processing dimensions include: batch, seq, expert
        # so, shape transformation of x: (batch, seq, d_input) -> (batch, seq, 1, 1, d_input)
        # two '1's:
        #    1st '1' is batch-processing dim 'expert', waiting for broadcasting (1 -> n_activate)
        #    2nd '1' is the dim for matmul like (1, d_input)@(d_input, d_hidden)
        x = x.unsqueeze(-2).unsqueeze(-3)

        # Step2: Top-K Sampling
        
        # use indices returned by torch.topk to do complex slicing on weight
        # shape changes of W1 & b1:
        #    W1: (n_expert, d_input, d_hidden) -> (batch, seq, n_activate, d_input, d_hidden)
        #    b1: (n_expert, 1, d_hidden) -> (batch, seq, n_activate, 1, d_hidden)
        _, activated_indices = torch.topk(input=affinity, k=self.n_activate, dim=-1)  # (batch, seq, n_activate)
        W1 = self.W1[activated_indices, ...]
        b1 = self.b1[activated_indices, ...]
        W2 = self.W2[activated_indices, ...]
        b2 = self.b2[activated_indices, ...]

        # Step3: FFN (Activated Experts Only)

        h = self.act1(x @ W1 + b1)           # -> (batch, seq, n_activate, 1, d_hidden)
        if self.dropout > 0:
            h = F.dropout(h, p=self.dropout)
        output = h @ W2 + b2                 # -> (batch, seq, n_activate, 1, d_output)
        return (output.squeeze(), activated_indices)


class DeepSeekMoE(nn.Module):
    """
    MoE for DeepSeek models
    
    NOTICE:
        SharedExperts and RoutedExperts deal with shape transforming differently,
        because the weight & bias they use differs in shape.
        
        SharedExperts use original weight & bias.
        e.g. 1st weight shape: (n_expert, d_input, d_hidden)
             corresponding input shape: (1, batch*seq, d_input)

        RoutedExperts use indexed weight & bias. (indices come from torch.topk)
        e.g. weight shape: (batch, seq, n_activate, d_input, d_hidden)
             corresponding input shape: (batch, seq, 1, 1, d_input)
    """
    def __init__(self,
                 d_model: int,
                 d_hidden: int,
                 n_shared_expert: int,
                 n_routed_expert: int,
                 n_routed_expert_activate: int,
                 dropout: float,
                 device: torch.device,
                 expert_bias: bool):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.n_shared_expert = n_shared_expert
        self.n_routed_expert = n_routed_expert
        self.n_routed_expert_activate = n_routed_expert_activate
        self.expert_bias = expert_bias

        self.norm = nn.RMSNorm(normalized_shape=d_model, device=device)
        self.shared_experts = SharedExperts(
            n_expert=n_shared_expert,
            d_input=d_model, d_output=d_model,
            d_hidden=d_hidden,
            dropout=dropout,
            device=device,
        )
        self.router = nn.Linear(d_model, n_routed_expert, bias=False, device=device)
        self.routed_experts = RoutedExperts(
            n_expert=n_routed_expert,
            n_activate=n_routed_expert_activate,
            d_input=d_model, d_output=d_model,
            d_hidden=d_hidden,
            dropout=dropout,
            device=device,
        )
        
        # Affinity Bias of DeepSeek-V3
        #
        # bias usage:
        #     Add the bias to (token-to-expert) affinity before topk-sampling.
        #     !! ONLY use bias for SAMPLING. !!
        # bias update:
        #     Increase an expert's bias when assigned tokens are less than the average.
        #     Otherwise, decrease it.
        if self.expert_bias: # to fit with affinity (batch, seq, n_expert)
            self.bias = torch.full((1, 1, n_routed_expert), fill_value = 1 / n_routed_expert, device=device,)
        else:
            self.bias = None

    def update_bias(self, u: float, c: torch.LongTensor):
        """
        Arguments:
            u: update rate
            c: assigned tokens count `(n_routed_expert,)`
        """
        assert self.bias is not None
        e = c.mean() - c                # load violation error
        self.bias += u * e              # broadcasting

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        inputs = self.norm(x)           # (batch, seq, d_model)
        shared_output = torch.sum(self.shared_experts(inputs), dim=0)  # sum over expert dim

        # Step 1: Routed Experts Forward

        raw_affinity = self.router(inputs)  # (batch, seq, n_expert)
        if self.expert_bias:
            assert self.bias is not None
            affinity = raw_affinity.detach() + self.bias
        else:
            affinity = raw_affinity
        batched_routed_output, activated_indices = self.routed_experts(inputs, affinity)

        # Step 2: Slicing the Top-K Affinity

        # slicing affinity score with expert indices
        # shape change (broadcasting): (..., n_routed_expert) -> (..., n_activate)
        slicing_index0 = torch.arange(batch).reshape(batch, 1, 1)   # [[[0]], [[1]], ...]
        slicing_index1 = torch.arange(seq).reshape(1, seq, 1)       # [[[0], [1], ...]]
        activated_affinity = raw_affinity[slicing_index0, slicing_index1, activated_indices].unsqueeze(-2)  # -> (batch, seq, 1, n_activate)
        
        # Step 3: matmul affinity & routed output
        
        # sum experts' output with matmul. DeepSeek-V3 introduced an additional normalization
        # (..., 1, n_activate) @ (..., n_activate, d_model) -> (batch, seq, 1, d_model)
        if self.expert_bias:
            normalized_affinity = activated_affinity / torch.sum(activated_affinity, dim=-1, keepdim=True)
            routed_output = (normalized_affinity @ batched_routed_output).squeeze()
        else:
            routed_output = (activated_affinity @ batched_routed_output).squeeze()

        return x + shared_output + routed_output

    @property
    def num_params(self) -> int:
        return (self.shared_experts.num_params
                + self.routed_experts.num_params
                + self.d_model * self.n_routed_expert)


