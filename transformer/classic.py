from .modules.ffn import DenseFFN
from .modules.attention import MultiHeadAttention
from .modules.positional_encoding import get_sinusoidal_pe_template, get_sinusoidal_pe

import torch
import torch.nn.functional as F
from torch import nn


class TransformerEncoder:
    """
    classic Transformer Encoder from *Attention is All You Need*
    
    `d_model // head = d_h = d_qk`
    """
    def __init__(self,
                 layers: int,
                 vocab_size: int,
                 d_model: int,
                 n_h: int,
                 d_qk: int,
                 d_v: int,
                 d_hidden_scale: int,
                 max_seq_len: int,
                 device: torch.device, dropout: float,):
        """
        Arguments:
            layers: number of Transformer blocks
            vocab_size: size of Tokenizer's vocabulary
            d_model: dim of word vector
            d_hidden_scale: hidden dim of FFN divides d_model (d_hidden / d_model)
            max_seq_len: max sequence length
        """
        self.layers = layers
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_h = n_h
        self.max_seq_len = max_seq_len
        self.d_h = d_model // n_h  # d_model // head = d_h = d_qk
        self.div_term_template, self.token_position_template = get_sinusoidal_pe_template(d_model=d_model, max_seq_len=max_seq_len, device=device,)
        
        self.embedding = nn.Linear(vocab_size, d_model, bias=False, device=device)
        self.attention_blocks: list[MultiHeadAttention] = []
        self.ffn_blocks: list[DenseFFN] = []
        for _ in range(layers):
            self.attention_blocks.append(
                MultiHeadAttention(
                    n_h=n_h, d_qk=d_qk, d_v=d_v,
                    d_model=d_model,
                    dropout=dropout, device=device,
                )
            )
            self.ffn_blocks.append(
                DenseFFN(
                    d_input=d_model, d_output=d_model,
                    d_hidden_size=d_hidden_scale * d_model,
                    activation=nn.ReLU(),  # Original Transformer use ReLU
                    dropout=dropout, device=device,
                )
            )

    def encode(self, prompt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        encode and return encoded tokens. (boolean mask rule: `True -> mask`)

        Arguments:
            prompt: tokenized sequence of shape (batch, seq, vocab_size)
            mask: boolean padding mask of shape (1, 1, d_model)
        """
        pe = get_sinusoidal_pe(seq_len=prompt.shape[1],
                               d_model=self.d_model,
                               token_position=self.token_position_template,
                               div_term=self.div_term_template)
        x = self.embedding(prompt) + pe
        for att, ffn in zip(self.attention_blocks, self.ffn_blocks):
            x = att(x, x, mask)  # q_input = kv_input = x
            x = ffn(x)
        return x

    @property
    def num_params(self) -> int:
        """number of parameters in Transformer encoder"""
        num_embedding = self.vocab_size * self.d_model
        att, ffn = self.attention_blocks[0], self.ffn_blocks[0]
        return num_embedding + self.layers * (att.num_params + ffn.num_params)


class TransformerDecoder:
    """classic Transformer Decoder from *Attention is All You Need*"""
    def __init__(self,
                 layers: int,
                 vocab_size: int,
                 d_model: int,
                 n_h: int,
                 d_qk: int,
                 d_v: int,
                 d_hidden_scale: int,
                 max_seq_len: int,
                 device: torch.device, dropout: float,):
        self.layers = layers
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_h = n_h
        self.d_qk = d_qk
        self.d_v = d_v
        self.max_seq_len = max_seq_len
        self.d_h = d_model // n_h  # commonly used by q,k,v
        self.div_term_template, self.token_position_template = get_sinusoidal_pe_template(d_model=d_model, max_seq_len=max_seq_len, device=device,)

        self.embedding = nn.Linear(vocab_size, d_model, bias=False, device=device)
        self.causal_masked_attention_blocks: list[MultiHeadAttention] = []
        self.cross_attention_blocks: list[MultiHeadAttention] = []
        self.ffn_blocks: list[DenseFFN] = []
        for _ in range(layers):
            self.causal_masked_attention_blocks.append(
                MultiHeadAttention(
                    n_h=self.n_h, d_qk=self.d_qk, d_v=self.d_v,
                    d_model=d_model,
                    dropout=dropout, device=device,
                )
            )
            self.cross_attention_blocks.append(
                MultiHeadAttention(
                    n_h=self.n_h, d_qk=self.d_qk, d_v=self.d_v,
                    d_model=d_model,
                    dropout=dropout, device=device,
                )
            )
            self.ffn_blocks.append(
                DenseFFN(
                    d_input=d_model, d_output=d_model,
                    d_hidden_size=d_hidden_scale * d_model,
                    activation=nn.ReLU(),  # Original Transformer use ReLU as activation
                    dropout=dropout, device=device,
                )
            )
        self.output = nn.Linear(d_model, vocab_size, bias=False, device=device)

    def decode(
        self,
        target: torch.Tensor,
        causal_mask: torch.Tensor,
        encoder_padding_mask: torch.Tensor,
        encoded_prompt: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode probability distribution for the next token with encoded_prompt and previously decoded sequence.

        Arguments:
            target: The decoded sequence.
            causal_mask: Mask for causal attention block.
                         Each decoded sequence in the batch have equal length (no padding mask).
            encoder_padding_mask: Mask the padding tokens in inputs prompts when input sequences differs in length.
            encoded_prompt: output of Transformer encoder
        """
        pe = get_sinusoidal_pe(
            seq_len=target.shape[1],
            d_model=self.d_model,
            token_position=self.token_position_template,
            div_term=self.div_term_template
        )
        x = self.embedding(target) + pe
        for causal_att, cross_att, ffn in zip(
            self.causal_masked_attention_blocks,
            self.cross_attention_blocks,
            self.ffn_blocks
        ):
            x = causal_att(x, x, causal_mask)  # q_input = kv_input = x
            x = cross_att(x, encoded_prompt, encoder_padding_mask)  # q_input = x, kv_input = encoded_prompt
            x = ffn(x)

        logits = self.output(x)
        return F.softmax(logits)

    @property
    def num_params(self) -> int:
        """number of parameters in Transformer decoder"""
        num_embedding = self.vocab_size * self.d_model
        num_prob_casting = self.d_model * self.vocab_size
        causal_att, cross_att = self.causal_masked_attention_blocks[0], self.cross_attention_blocks[0]
        ffn = self.ffn_blocks[0]
        return num_embedding + self.layers * (causal_att.num_params + cross_att.num_params + ffn.num_params) + num_prob_casting


class Transformer:
    """
    Transformer according to Attention is All You Need
    https://arxiv.org/abs/1706.03762
    """
    def __init__(self,
                 d_model: int,
                 n_h: int,
                 d_qk: int,
                 d_v: int,
                 d_hidden_scale: int,
                 encoder_layers: int,
                 encoder_vocab_size: int,
                 encoder_max_seq_len: int, 
                 decoder_layers: int,
                 decoder_vocab_size: int,
                 decoder_max_seq_len: int,
                 device: torch.device,
                 dropout: float,):
        self.encoder = TransformerEncoder(
            layers=encoder_layers,
            vocab_size=encoder_vocab_size,
            d_model=d_model,
            n_h=n_h, d_qk=d_qk, d_v=d_v,
            d_hidden_scale=d_hidden_scale,
            max_seq_len=encoder_max_seq_len,
            device=device, dropout=dropout,
        )
        self.decoder = TransformerDecoder(
            layers=decoder_layers,
            vocab_size=decoder_vocab_size,
            d_model=d_model,
            n_h=n_h, d_qk=d_qk, d_v=d_v,
            d_hidden_scale=d_hidden_scale,
            max_seq_len=decoder_max_seq_len,
            device=device, dropout=dropout,
        )
    
    @property
    def num_params(self) -> int:
        return self.encoder.num_params + self.decoder.num_params

    def get_causal_mask(self, dim: int):
        """
        boolean mask rule: `True -> mask`.
        return shape `(1, 1, seq, seq)`, broadcasting along batch&head dim
        """
        inversed_mask = torch.tril(torch.ones((dim, dim), dtype=torch.bool))  # False -> mask
        return ~inversed_mask  # True -> mask

    def encode(self, prompt, padding_mask) -> torch.Tensor:
        return self.encoder.encode(prompt, padding_mask)

    def decode(self, target, causal_mask, encoder_padding_mask, encoded_prompt) -> torch.Tensor:
        return self.decoder.decode(target, causal_mask, encoder_padding_mask, encoded_prompt)
