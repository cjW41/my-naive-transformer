# Naively Replicated Transformers

## 1. Current Works
- `kv_cache_test`
    - [**deprecated**] kv-cache for classic Transformer. waiting for merging into `transformer`.
- `transformer`
    - [**debugging**] the classic Transformer according to [*Attention is All You Need*](https://arxiv.org/abs/1706.03762)
    - [**debugging**] DeepSeek-V2 according to [*DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*](https://arxiv.org/abs/2405.04434)

## 2. Modules
- `.modules`: building blocks
    - `.attention`: Attention functions and modules for Transformer.
    - `.ffn`: FFN and MoE modules.
    - `.positional_encoding`: Functions for generating positional encoding.
- `.deepseek`: DeepSeek family
- `.classic`: Classic models like BERT, Transformer.

## 3. TODO List
1. rebuild kv_cache modules in `transformer.module`
2. add `DeepSeek-V3`, `DeepSeek-V3.2` to `transformer.deepseek`
3. replicate `Qwen` model familiy
