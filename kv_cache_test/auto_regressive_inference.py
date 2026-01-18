"""
Inference tools:
    - Tokenizer
    - AutoRegressiveInference
"""


from utils import TokenizerBase, MyDevice, Const, KVCache
from basic_transformer import MyTransformer

import torch
from torch import Tensor
from torch.nn.functional import softmax

from typing import Optional, List, Union
from collections import defaultdict


class Tokenizer(TokenizerBase):

    special_tokens = ['[PAD]',  # padding
                      '[UNK]',  # unknown vocabulary
                      '[SOS]',  # start of sequence
                      '[EOS]']  # end of sequence

    def __init__(self,
                 max_dict_size: int,
                 max_seq_len: int,
                 text_to_fit: Optional[List[str]],
                 addional_special_tokens: Optional[List[str]] = None):
        """A trivial implementation of Tokenizer. 
        Language: ASSUMING that input and output are using the same language and one common dictionary.
        Tokenizing: Segment texts into sequence of words. Index words by rank of frequency.
        Truncating: Discard the trailing part exceeding the max-sentence-length limit.
        """
        super().__init__()  # word2idx, idx2word
        self.max_dict_size = max_dict_size
        self.dict_size = -1  # the real dict size initialized in _fit()
        self.max_seq_len = max_seq_len
        if text_to_fit is not None:
            self._fit(text_to_fit)
        if addional_special_tokens is not None:
            self.special_tokens.extend(addional_special_tokens)

    @property
    def dict_volume(self):
        #  Maximum number of words the dictionary can carry.
        return self.max_dict_size - len(self.special_tokens)

    @property
    def max_sentence_len(self):
        return self.max_seq_len - 2  # SOS&EOS

    def encode(self, texts: List[str], to_device: MyDevice) -> Tensor:
        """translate words to tokens

        Args:
            texts (List[str]): A list of texts. One list component is a sentence.
            to_device (MyDevice): Return the tensor to targeted device.

        Returns:
            tokenized texts of shape:(batch, seq)
        """
        # Step1: Truncate and Pad
        # truncate by discarding the words exceeding max-lenght limit
        segmented_texts = [self._simple_word_segment(text)
                           for text in texts]
        del texts
        segmented_texts = [text[:self.max_sentence_len]
                           for text in segmented_texts]  # [seq1_list,...]
        # add SOS,EOS and pad
        pad_template = ['[PAD]'] * self.max_sentence_len
        for text in segmented_texts:
            text.insert(0, '[SOS]')
            text.append('[EOS]')
            vacency = self.max_sentence_len - len(text)
            if vacency > 0:
                text.extend(pad_template[:vacency])
        del pad_template

        # Step2: Transform Word to Idx
        idx_texts = []
        for text in segmented_texts: # list of word -> list of idx
            idx_text = []
            for token in text:
                idx_text.append(self.encode_word(token))
            idx_texts.append(idx_text)

        return torch.tensor(idx_texts).to(device=to_device.value)
    
    def decode(self, texts: Tensor) -> List[List[str]]:
        """translate tokens to words

        Args:
            texts (Tensor]): The generated text of tokens. shape:(batch,seq)
        
        Returns:
            Translated list of sentences. One component of outer/inner list is one sentence/word. 
        """
        token_filter = [self.word2idx['[PAD]'],
                        self.word2idx['[SOS]'],
                        self.word2idx['[EOS]']]

        word_texts = []
        text_idxs = range(texts.size(0))
        token_idxs = range(texts.size(1))

        for text_idx in text_idxs:
            word_text = []
            for token_idx in token_idxs:
                token = texts[text_idx, token_idx].item()
                if token not in token_filter:  # remove meaningless tokens
                    assert isinstance(token, int)
                    word_text.append(self.decode_word(token))
            word_texts.append(word_text)
        
        return word_texts

    def encode_word(self, word: str) -> int:
        #  return idx of UNK when word does not exist
        return self.word2idx.get(word, self.word2idx['[UNK]'])

    def decode_word(self, idx: int) -> str:
        #  return UNK when idx does not exist
        return self.idx2word.get(idx, '[UNK]')

    def _fit(self, texts: List[str]) -> None:
        # word counting
        wordfreq = defaultdict(int)
        for text in texts:
            word_list = self._simple_word_segment(text)
            for word in word_list:
                wordfreq[word] += 1
        #  ranking by count
        #  ranked_word: [(word_rank1, count1),...]
        ranked_word = sorted(wordfreq.items(),
                             key=lambda item:item[1], # rank by value of dict item:(key, val)
                             reverse=True)
        #  truncate the ranked word list according to dict volume, index the remaining words
        truncated_word_list = [word for word, _ in ranked_word][:self.dict_volume]

        # real size of dictionary
        self.dict_size = len(truncated_word_list)

        idx = 0
        for word in truncated_word_list:
            self._add(idx, word)
            idx += 1
        for special_token in self.special_tokens:
            self._add(idx, special_token)
            idx += 1
        #  buffer the frequently used tokens
        self.pad_idx = self.word2idx['[PAD]']
        self.unk_idx = self.word2idx['[UNK]']
    
    def _add(self, idx, word):
        self.word2idx[word] = idx
        self.idx2word[idx] = word

    @staticmethod
    def _simple_word_segment(text: str):
        return text.lower().split()


class AutoRegressiveInference:

    max_inference_step: int = 500
    top_k: int = 50
    top_p: float = 0.9

    def __init__(self,
                 model: MyTransformer,
                 tokenizer: TokenizerBase,
                 mydevice: MyDevice,
                 dtype: torch.dtype,
                 max_inference_step: Optional[int],
                 top_k: Optional[int],
                 top_p: Optional[float]):
        """Auto-regressive inference of BasicTransformer. 

        Args:
            model (Module): the language model
            tokenizer (TokenizerBase): transform(encode) input text to symbolic representation(index)
            max_output_length (Optional[int]): max output length including special tokens
        """
        self.model = model.to(mydevice.value)
        self.model.eval()
        self.num_stack=self.model.num_stack
        self.heads=self.model.heads
        self.d_kv=self.model.d_qkv

        self.tokenizer = tokenizer
        self.mydevice = mydevice
        self.dtype = dtype
        if max_inference_step and max_inference_step > 0:
            self.max_inference_step = max_inference_step
        if top_k and 0 < top_k:
            self.top_k = top_k
        if top_p and 0 < top_p < 1:
            self.top_p = top_p

        self.causal_mask = torch.triu(torch.ones(self.max_inference_step, self.max_inference_step))  # upper triangle matrix

        self.padding_idx = self.tokenizer.word2idx['[PAD]']
        self.sos_idx = self.tokenizer.word2idx['[SOS]']
        self.eos_idx = self.tokenizer.word2idx['[EOS]']

    def inference(self, prompt: List[str], temperature: float = 1., return_words: bool = False):
        """**Auto-Regressive Inference**

        Each time, ONLY the token generated in the last step is passed to decoder.
        The predicted token is then concated with the earlier generated sequence, and passed to decoder in the new step.

        Args:
            prompt (List[str]): Raw input sentence
            temperature (float): Scaling parameter of logits. Defaults to 1.
            return_words (bool): Return list of tokens or list of words. Defaults to False (tokens).

        Returns:
            the generated tokens or words in a list.
        """
        # ! In this function, batched inference is disabled.
        # ! To implement batched inference, one can implement either one of the following strategies.
        # !     1. Terminated sample (together with cache) should be stored elsewhere and deleted from target.
        # !        Inference temrinated when all samples are removed. 
        # !     2. Store the terminated sample and implement a whole-sequence mask.
        # !        Inference terminated when all samples are masked.
        assert len(prompt) == 1
        assert temperature > 0
        
        # Step1: Prefix (Prefill Phase)
        # tokenize, encode
        prompt_idx = self.tokenizer.encode(prompt, self.mydevice)
        batch, prompt_len = prompt_idx.size(dim=0), prompt_idx.size(dim=1)
        prompt_mask = self._padding_mask_generate(prompt_idx)
        encoded_prompt = self.model.encode(prompt_idx, prompt_mask)

        # cross attention kv cache initialize
        cross_k_cache = torch.empty(self.num_stack, batch, self.heads, prompt_len, self.d_kv,
                                         device=torch.device(self.mydevice.value), dtype=self.dtype)
        cross_v_cache = torch.clone(cross_k_cache)
        for layer_id, decoder_layer in enumerate(self.model.decoder.module_list):
            # Compute kv. Eventual kv shape:(batch, head, seq, kv)
            W_k = decoder_layer.cross_att.W_k
            k = W_k(encoded_prompt).reshape(batch, prompt_len, self.heads, self.d_kv).transpose(1, 2)
            W_v = decoder_layer.cross_att.W_v
            v = W_v(encoded_prompt).reshape(batch, prompt_len, self.heads, self.d_kv).transpose(1, 2)
            # Insert kv to cache.
            cross_k_cache[layer_id], cross_v_cache[layer_id] = k, v  

        # kv cache object initialize
        kvcache = KVCache(num_stack=self.model.num_stack,
                          batch=batch,
                          heads=self.model.heads,
                          max_output_len=self.max_inference_step + 1,  # +1 stands for the [SOS] token
                          d_kv=self.model.d_qkv,
                          mydevice=self.mydevice,
                          dtype=self.dtype,
                          cross_k_cache=cross_k_cache,
                          cross_v_cache=cross_v_cache)

        # Step2: Decode
        # With KV Cache, due to the mechanism of causal mask, only the token generated in the last step is passed to decoder.
        # Output of decoder is exactly the logits for the next token.
        target = torch.tensor([self.sos_idx]*batch).unsqueeze(dim=1)  # input of decoder in each step
        generated = torch.clone(target)  # the whole output of decoder
        for step in range(self.max_inference_step):
            causal_mask = self._causal_mask_generate(step)
            logits = self.model(target, causal_mask, kvcache) / temperature
            next_token = self._filter_and_sample(logits)
            torch.cat((generated, next_token), dim=1, out=target)
            if target[0, 0] == self.eos_idx:  # terminate the inference
                break
            else:
                target = next_token  # update target and commence the next step

        if return_words:
            return self.tokenizer.decode(generated)
        else:
            return generated

    def _filter_and_sample(self, logits: Tensor) -> Tensor:
        """Implement top-p and top-k filter on Logits, then sample from it.
        Returns:
            next_token: symbolic representation of the next token of each batch.
        """
        sorted_probs, sorted_indices = torch.sort(softmax(logits, dim=-1), descending=True, dim=-1)
        # top-p filter
        # Notice:
        #   If the 1st prob>top_p, according to the filtering rule, no prob will be saved.
        #   AT LEAST one prob should be saved, hence the 1st sorted prob is always saved.
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        top_p_filter = cumulative_probs > self.top_p  # True->discard, False->save
        top_p_filter[:, 0] = False  # always save the 1st prob
        # top-k filter
        top_k_filter = torch.ones_like(logits).to(dtype=torch.bool)
        top_k_filter[:,:self.top_k] = False  # save top_k tokens

        # mask the logits
        combined_filter = top_k_filter | top_p_filter
        logit_mask = torch.ones_like(logits).to(dtype=torch.bool)  # initialize
        logit_mask = logit_mask.scatter_(dim=-1,  # scatter along logit dim
                                         index=sorted_indices,  # sorted_indices serve as index of logit dim
                                         src=combined_filter)  # fill True to the component to be masked
        masked_logits = logits.masked_fill(logit_mask, Const.NegInf)
        del logits

        # sampling
        probs = softmax(masked_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (batch,1)
        return next_tokens  # shape:(batch,token_idx=1)
    
    def _padding_mask_generate(self, prompt_idx: Tensor) -> Tensor:
        """Add padding mask to attention scores, return the result.

        Args:
            prompt_idx (Tensor): prompt in symbolic representation (index). shape:(batch, seq)
        
        Example:
            ([SOS]=100,[EOS]=101,[PAD]=102) prompt_idx = [[100,1,2,101,102,102],[1000,3,4,5,101,102]]
            -> ([PAD]=0,other=1) padding mask = [[1,1,1,1,0,0],[1,1,1,1,1,0]]
        """
        return (prompt_idx != self.padding_idx).unsqueeze(dim=1).unsqueeze(dim=1)  # add two dim: (batch, seq) -> (batch, 1, 1, seq)

    def _causal_mask_generate(self, step: int) -> Tensor:
        return self.causal_mask[:step+1, :step+1]





