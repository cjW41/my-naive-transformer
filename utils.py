"""
Const: constant container
MyDevice: device name container
EncoderDecoder: base class of encoder-decoder architectural model
TokenizerBase: base class of Tokenizer
StackOfModule: base class of encoder/decoder
KVCache: kv cache for masked&cross attention in decoder
"""


import torch
from torch import Tensor
from torch.nn import Module

from typing import List, Dict, Tuple
from abc import ABC, abstractmethod
from enum import Enum


class Const:
    Eps = 1e-9
    NegInf = -1e9


class MyDevice(Enum):
    CPU = 'cpu'
    GPU = 'cuda'


class EncoderDecoder:
    
    def __init__(self,
                 encoder: Module,
                 decoder: Module,
                 freeze_encoder: bool):
        """fundamental encoder-decoder architecture. input & output of encoder & decoder are of same shape:
        (batch, seq, word vector)

        Args:
            encoder (Module): encoder.forward() should take *prompt, prompt_mask* as input
            decoder (Module): decoder.forward() should take *target, target_mask, encoded_prompt, prompt_mask* as input
            freeze_encoder (bool): When encoder is freezed, it will not participate in backpropagation.
        """
        self.encoder = encoder
        self.decoder = decoder
        self.freeze_encoder = freeze_encoder
        self.encoded_prompt = None
        self.prompt_mask = None

    def encode(self, prompt: Tensor, prompt_mask: Tensor):
        """Encode the tokenlized prompt.

        Args:
            prompt (Tensor): tokenlized prompt. shape:(batch, seq)
            prompt_mask (Tensor): The padding mask truncating the trailing [PAD]s.
        """
        if self.freeze_encoder:
            with torch.no_grad():
                self.encoded_prompt = self.encoder(prompt, prompt_mask)
        else:  # generate encoded prompt with computational map for the backpropagation
            self.encoded_prompt = self.encoder(prompt, prompt_mask)
        self.prompt_mask = prompt_mask
        
        return self.encoded_prompt
    
    def decode(self, target: Tensor, target_mask: Tensor, kvcache: "KVCache") -> Tensor:
        """predict distribution of the next word in target by applying softmax to output of decoder.

        Args:
            target (Tensor): current generated sentence. Dimension of target is same as prompt.
            target_mask (Tensor): The causal mask implemented on target in decoder.
        """
        return self.decoder(target, target_mask, self.encoded_prompt, self.prompt_mask, kvcache)

    def forward(self, x):
        pass


class TokenizerBase(ABC):

    def __init__(self) -> None:
        super().__init__()
        self.word2idx: Dict[str, int] = dict()
        self.idx2word = dict()

    @abstractmethod
    def encode(self, texts: List[str], to_device: MyDevice) -> Tensor:
        """translate words to tokens

        Args:
            texts (List[str]): A list of texts. One list component is a sentence.
            to_device (MyDevice): Return the tensor to targeted device.

        Returns:
            tokenized texts of shape:(batch, seq)
        """
        pass

    @abstractmethod
    def decode(self, texts: Tensor) -> List[List[str]]:
        """translate tokens to words

        Args:
            texts (Tensor]): The generated text of tokens. shape:(batch,seq)
        
        Returns:
            Translated list of sentences. One component of outer/inner list is one sentence/word. 
        """
        pass
    
    @abstractmethod
    def _fit(self, texts: List[str]) -> None:
        """training the tokenizer with the input texts.
        This method is only called in __init__().
        """
        pass


class StackOfModule(Module):

    def __init__(self,
                 num_stack: int,
                 module_class: type,
                 **initialize_kwargs):
        """Initialize a stack of _num_ component (encoder/decoder) modules.
        The class passed in should be the module with same shape of input&output.

        Args:
            num_stack (int): num of modules.
            module_class (type): class of the component module
            **initialize_kwargs: args to initialize each component module
        """
        super().__init__()
        self.module_list = list()
        for stack_idx in range(num_stack):
            self.module_list.append(module_class(stack_idx, **initialize_kwargs))

    def forward(self, x, **forward_kwargs):
        """
        Args:
            x (Tensor): inputs of the first module
            **forward_kwargs: args passed to forward() of each component module
        """
        for module in self.module_list:
            x = module(x, **forward_kwargs)
        return x


class KVCache:

    def __init__(self,
                 num_stack: int, batch: int, heads: int, max_output_len: int, d_kv: int,
                 mydevice: MyDevice, dtype: torch.dtype,
                 cross_k_cache: Tensor, cross_v_cache: Tensor):
        """KV Cache for encoder-decoder model. shape:(layer, batch, head, seq, kv)

        Args:
            num_stack (int): number of layers of Decoder
            batch (int): batch size
            heads (int): number of attention heads
            max_output_len (int): max output sequence length (includes special tokens)
            d_kv (int): dimension of k,v
            cross_k_cache, cross_v_cache: kv cache for cross attention blocks computed elsewhere
        """
        self.max_output_len = max_output_len
        self.cross_k_cache, self.cross_v_cache = cross_k_cache, cross_v_cache   
        # initialize the whole kv cache for masked attention in memory in case of memory running out.
        self.current_filled_num = 0  # +1 when one pair of kv is added to cache
        self.masked_k_cache = torch.empty(num_stack, batch, heads, max_output_len-1, d_kv,
                                          device=torch.device(mydevice.value), dtype=dtype)
        self.masked_v_cache = torch.empty_like(self.masked_k_cache)

    @property
    def masked_cache_full(self) -> bool:
        return self.current_filled_num == self.max_output_len - 1  # full -> True

    @property
    def masked_cache(self) -> Tuple[Tensor, Tensor]:
        return (self.masked_k_cache[:, :, :, :self.current_filled_num, :],
                self.masked_v_cache[:, :, :, :self.current_filled_num, :])

    @property
    def cross_cache(self) -> Tuple[Tensor, Tensor]:
        return self.cross_k_cache, self.cross_v_cache

    def update_masked_cache(self, stack_idx: int , new_k: Tensor, new_v: Tensor):
        """Insert k,v of one layer (of decoder) to cache.
        
        Args:
            stack_idx (int): index of the layer to insert k,v
            new_k, new_v (Tensor): cache kv of a new token passed to decoder. shape:(batch, heads, seq=1, kv)
        """
        next_idx_in_seq = self.current_filled_num
        self.masked_k_cache[stack_idx, :, :, next_idx_in_seq, :] = new_k.squeeze()  # remove the seq dim to fit in the cache
        self.masked_v_cache[stack_idx, :, :, next_idx_in_seq, :] = new_v.squeeze()
        self.current_filled_num += 1



