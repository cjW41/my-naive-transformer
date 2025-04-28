# coding:utf-8
"""
Foundamental Version of Auto-Regressive Transformer:
    - Transformer:                  Transformer class based on 'Attention is All u Need'.
    - Encoder, Decoder:             Stack of encoder & decoder blocks.
    - EncoderBlock, DecoderBlock:   Unit module of encoder & decoder.
    - LayerNormalization, FFN
    - MultiHeadBase:                Base class for multi-head attention.
    - EncoderMultiHeadAtt:          Self attention in encoder block.
    - CrossMultiHeadAtt:            Cross attention in decoder block.
    - MaskedMultiHeadAtt:           Masked self attention in decoder block.

shape of tensor:
    - batch:        size of batch
    - seq:          length of sequence
    - word vector:  =d_model, length of word vector   
    - dict_size:    size of dictionary
    - qkv:          length of q,k,v
    - heads:        number of heads in multi-head attention
"""


from utils import Const, EncoderDecoder, StackOfModule, KVCache

import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor
from torch.nn.functional import log_softmax, dropout

from typing import Callable
import math


class MyTransformer(Module, EncoderDecoder):
    
    def __init__(self,
                 embedder: "Embedder",
                 target_dict_size: int,
                 num_stack: int = 10,
                 d_model:int =512,
                 heads:int = 8,
                 dropout_rate:float = 0.1,
                 freeze_encoder: bool = False):
        """Basic implementation of the auto-regressive Transformer based on *Attention is All You Need*.<br>
        
        Args:
            embedder (Embedder): Embed the symbolic representation of prompt and add positional information.
            target_dict_size (int): size of target dictionary.
            num_stack (int): Number of unit modules in encoder/decoder.
            d_model (int): Dimension of model, aka dimension of word vector, should be power of 2. Defaults to 512.
            heads (int): Num of heads in multi-head attention. d_model % heads = 0. Defaults to 8.
            dropout_rate (float): The dropout rate used in attention function
            freeze_encoder (bool): When encoder is freezed, it will not participate in backpropagation.
        """
        assert d_model % 2 == 0  # dimension of model is even (positional encoding)
        assert d_model % heads == 0  # length of q,k vector is d_model/heads (multi-head attention)
        self.d_qkv = d_model // heads

        super().__init__(encoder=Encoder(num_stack, d_model, self.d_qkv, dropout_rate),
                         decoder=Decoder(num_stack, d_model, self.d_qkv, dropout_rate),
                         freeze_encoder=freeze_encoder)  # args for EncoderDecoder
        
        self.num_stack = num_stack
        self.d_model = d_model
        self.heads = heads
        # Notice:
        # Theoretically, d_v is free of d_q,d_v. For convenience, d_v is normally set to match
        # the equation d_v*heads=d_model so that shape of output equals to the shape of input
        # in multi-head attention. Therefore, normally d_q=d_k=d_v=d_qkv.

        self.embedder = embedder  # prompt and target use the same embedder

        #  shape:(batch, seq, d_model) -> (batch, seq, target_dict_size)
        self.projection = nn.Linear(d_model, target_dict_size)

    def forward(self, target: Tensor, target_mask: Tensor, kvcache: KVCache):
        """
        Args:
            target (Tensor): current generated sentence. Dimension of target is same as prompt.
            target_mask (Tensor): causal mask of output sequence
            **kv_cache: k,v cache fetched from a KVCache object

        Returns:
            logits: shape(batch, dict size)
        """
        # encoding check
        if self.encoded_prompt is None:
            raise ValueError('please encode the prompt before calling forward()')
        # decode
        decoder_output = self.decode(self.embedder(target), target_mask, kvcache)
        # projection: decoder_output -> Logits
        logits = self.projection(decoder_output)
        return logits


class Embedder(Module):

    def __init__(self,
                 word_embedder: Module,
                 embed_ratio: float,
                 positional_encode_generate: Callable,
                 **kwargs_pe):
        """Embed the preproccessed tokenized prompt/target (symbolic representation)

        Args:
            word_embedder (Module): token->word vector by a trainable parameter.
            embed_ratio (float): scaling parameter of word embedding, usually
            positional_encode_generate (Callable): add positional encoding template to module buffer
        """
        super().__init__()
        self.word_embedder = word_embedder
        self.embed_ratio = embed_ratio
        # positional encoding buffer
        # pe shape:(1, seq, word vector)
        self.register_buffer('pe', positional_encode_generate(**kwargs_pe))

    def forward(self, inputs: Tensor):
        """
        Args:
            inputs (Tensor): tokenized prompt/target. shape:(batch, seq).
        
        Return:
            embedded prompt/target in continuous representation. shape:(batch, seq, word vector)
        """
        inputs_seq_len = inputs.size(-2)
        embedded = self.word_embedder(inputs) * self.embed_ratio
        torch.add(embedded, self.pe[:, :, :inputs_seq_len], out=embedded)
        return embedded
    

def sin_positional_encode(d_model: int, seq_len: int):
    """Foundamental Sin positional encoder.
    Return:
        positional code of shape: (1, seq, word vector)
    """
    # position of word in sequence
    pos_along_seq = torch.arange(seq_len).unsqueeze(1)  # row to column vector
    # id of component in word vector: 1 / 10000^(i/d_model) = e^(i/(-d_model/log(10000)))
    factor_along_word_vec = torch.exp(torch.arange((d_model // 2)) / ( - d_model/math.log(10000)))  # row vector
    
    pos_code = torch.empty(seq_len, d_model)
    pos_code[:, 0::2] = torch.sin(pos_along_seq * factor_along_word_vec)
    pos_code[:, 1::2] = torch.cos(pos_along_seq * factor_along_word_vec)
    pos_code.requires_grad_(False)
    return pos_code.unsqueeze(0)  # (seq, word vector) -> (batch, seq, word vector)


class Encoder(StackOfModule):
    """Encoder of BasicTransformer composed of _num_stack_ encoder block"""
    def __init__(self,
                 num_stack: int,
                 d_model: int, d_qkv: int,
                 dropout_rate: float):
        super().__init__(num_stack, EncoderBlock,
                         #  initialize_kwargs
                         d_model=d_model,
                         d_qkv=d_qkv,
                         dropout_rate=dropout_rate)

    def forward(self, prompt, prompt_mask):
        return super().forward(prompt,
                               prompt_mask=prompt_mask)  # forward_kwargs


class Decoder(StackOfModule):
    """Decoder of BasicTransformer composed of _num_stack_ decoder block"""
    def __init__(self,
                 num_stack: int, 
                 d_model: int,
                 d_qkv: int,
                 dropout_rate: float):
        super().__init__(num_stack, DecoderBlock,
                         #  initialize_kwargs
                         d_model = d_model,
                         d_qkv = d_qkv,
                         dropout_rate = dropout_rate)

    def forward(self, target, target_mask, encoded_prompt, prompt_mask, kvcache):
        #  kv_cache includes: k_cache, v_cache
        return super().forward(target,
                               #  forward_kwargs
                               target_mask=target_mask,
                               encoded_prompt=encoded_prompt,
                               prompt_mask=prompt_mask,
                               kvcache=kvcache)


class EncoderBlock(Module):
    
    def __init__(self, stack_idx: int,
                 #  initialize_kwargs
                 d_model: int,
                 d_qkv: int,
                 dropout_rate: float):
        super().__init__()
        #  inputs self attention
        self.multi_head_att = EncoderMultiHeadAtt(stack_idx, d_model, d_qkv, dropout_rate)
        self.layer_norm1 = LayerNormalization(d_model)
        self.ffn = FFN(d_model)
        self.layer_norm2 = LayerNormalization(d_model)

    def forward(self, x, prompt_mask):

        x = self.layer_norm1(x + self.multi_head_att(x, x, x, prompt_mask))
        return self.layer_norm2(x + self.ffn(x))


class DecoderBlock(Module):
    
    def __init__(self, stack_idx: int,
                 #  initialize_kwargs
                 d_model: int,
                 d_qkv: int,
                 dropout_rate: float):
        super().__init__()
        #  target self attention
        self.masked_att = MaskedMultiHeadAttention(stack_idx, d_model, d_qkv, dropout_rate)
        self.layer_norm1 = LayerNormalization(d_model)
        #  encoded input & target cross attention
        self.cross_att = CrossMultiHeadAttention(stack_idx, d_model, d_qkv, dropout_rate)
        self.layer_norm2 = LayerNormalization(d_model)
        #  FFN
        self.ffn = FFN(d_model)
        self.layer_norm3 = LayerNormalization(d_model)

    def forward(self, x, target_mask, encoded_prompt, prompt_mask, kvcache):
        x = self.layer_norm1(x + self.masked_att(x, x, x, target_mask, kvcache))
        x = self.layer_norm2(x + self.cross_att(x, encoded_prompt, encoded_prompt, prompt_mask, kvcache))
        return self.layer_norm3(x + self.ffn(x))


class LayerNormalization(Module):

    def __init__(self, layer_size: int) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.ones(layer_size))
        self.b = nn.Parameter(torch.zeros(layer_size))

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True, unbiased=False)  # unbiased variance
        return self.a * (x-mean) / (std+Const.Eps) + self.b
    

class FFN(Module):
    
    def __init__(self, d_model: int):
        super().__init__()
        ffn = nn.Sequential()
        ffn.append(nn.Linear(d_model, 8*d_model))  # hidden layer size defaults to 4096
        ffn.append(nn.ReLU())
        ffn.append(nn.Linear(8 * d_model, d_model))
        self.ffn = ffn
    
    def forward(self, x):
        return self.ffn(x)


class MultiHeadBase(Module):

    def __init__(self,
                 stack_idx: int,
                 d_model: int,
                 d_qkv: int,
                 dropout_rate: float):
        """Base class of multi-head attention.
        Args:
            stack_idx (int): index of block in StackOfMudle to be passed to kvcache.
            d_model (int): dimension of model = heads * d_qkv
            d_qkv (int): This module ASSUMES that q,k,v have same length.
        """
        assert d_model % d_qkv == 0
        super().__init__()
        self.stack_idx = stack_idx
        self.d_qkv = d_qkv
        self.heads = d_model // d_qkv
        self.dropout_rate = dropout_rate

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
    
    def forward(self, x_q: Tensor, x_k: Tensor, x_v: Tensor, mask: Tensor, kvcache: KVCache):
        """forward (with KV Cache in decoder).

        Args:
            x_q,x_k,x_v (Tensor): Input of qkv channels. shape:(batch, seq, word vector)
            mask (Tensor): causal mask or padding mask
        """
        raise NotImplementedError('forward() of multi-head attention not implemented')

    def attention_fn(self,
                     q: Tensor, k: Tensor, v: Tensor, mask: Tensor,
                     batch: int, q_seq: int):
        """calculate scaled dot-product attention, then output in shape:(batch, q_seq, word vector)"""
        # Notice: seq length of q does NOT equal to seq length of kv in CROSS ATTENTION.
        # shape
        #   q: (batch, head, q_seq, qkv); k^T: (batch, head, qkv, kv_seq)
        #   -> q*k^T: (batch, head, q_seq, kv_seq)
        score = log_softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_qkv),
                            dim=-1)

        # dropout and mask the score
        if self.dropout_rate > 0:
            score = dropout(score, self.dropout_rate)
        score = score.masked_fill(mask == 0, Const.NegInf)

        att_output = torch.matmul(score, v)  # shape:(batch, head, q_seq, qkv)
        #  (batch, head, q_seq, qkv) -> (batch, q_seq, head, qkv) -> (batch, q_seq, word vector)
        return att_output.transpose(1, 2).reshape(batch, q_seq, -1)


class EncoderMultiHeadAtt(MultiHeadBase):
    def __init__(self,
                 stack_idx: int,
                 d_model: int,
                 d_qkv: int,
                 dropout_rate: float):
        super().__init__(stack_idx, d_model, d_qkv, dropout_rate)

    def forward(self, x_q: Tensor, x_k: Tensor, x_v: Tensor, mask: Tensor):
        batch, seq = x_q.size(0), x_q.size(1)  # qkv have same sequence length
        # qkv
        q, k, v = self.W_q(x_q), self.W_k(x_k), self.W_v(x_v)
        # Reshape to cut last dim to heads number of slices. Exchange heads dim and sequence dim.
        #   (batch, seq, word vector) -> (batch, seq, head, qkv) -> (batch, head, seq, qkv)
        q = q.reshape(batch, seq, self.heads, self.d_qkv).transpose(1, 2)
        k = k.reshape(batch, seq, self.heads, self.d_qkv).transpose(1, 2)
        v = v.reshape(batch, seq, self.heads, self.d_qkv).transpose(1, 2)

        return self.attention_fn(q, k, v, mask, batch, seq)


class CrossMultiHeadAttention(MultiHeadBase):
    def __init__(self,
                 stack_idx: int,
                 d_model: int,
                 d_qkv: int,
                 dropout_rate: float):
        """cross multi-head attention in decoder (2nd attention block in decoder)"""
        super().__init__(stack_idx, d_model, d_qkv, dropout_rate)

    def forward(self, x_q: Tensor, mask: Tensor, kvcache: KVCache):
        """kv are stored in KVCache object, only x_q is needed."""
        # q
        batch, seq = x_q.size(0), x_q.size(1)
        q = self.W_q(x_q).reshape(batch, seq, self.heads, self.d_qkv).transpose(1, 2)
        # kv
        k, v = kvcache.cross_cache

        return self.attention_fn(q, k, v, mask, batch, seq)


class MaskedMultiHeadAttention(MultiHeadBase):
    def __init__(self,
                 stack_idx: int,
                 d_model: int,
                 d_qkv: int,
                 dropout_rate: float):
        super().__init__(stack_idx, d_model, d_qkv, dropout_rate)

    def forward(self, x_q: Tensor, x_k: Tensor, x_v: Tensor, mask: Tensor, kvcache: KVCache):
        # q
        batch, q_seq = x_q.size(0), x_q.size(1)
        q = self.W_q(x_q).reshape(batch, q_seq, self.heads, self.d_qkv).transpose(1, 2)

        # kv
        # new kv generated by the newest token
        new_k = self.W_k(x_k[:, -1, :]).unsqueeze(dim=1)  # shape:(batch,k)->(batch,seq=1,k)
        new_v = self.W_v(x_v[:, -1, :]).unsqueeze(dim=1)
        new_k = new_k.reshape(batch, 1, self.heads, self.d_qkv).transpose(1, 2)
        new_v = new_v.reshape(batch, 1, self.heads, self.d_qkv).transpose(1, 2)
        # update then fetch
        kvcache.update_masked_cache(self.stack_idx, new_k, new_v)
        k, v = kvcache.masked_cache

        return self.attention_fn(q, k, v, mask, batch, q_seq)



