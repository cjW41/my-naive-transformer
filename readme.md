# Some Naive Replication of Transformer

## 1. Current Works
- `transformer`
    - the classic Transformer according to [*Attention is All You Need*](https://arxiv.org/abs/1706.03762)
    - DeepSeek-V2 according to [*DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*](https://arxiv.org/abs/2405.04434)
    - DeepSeek-V3 according to [*DeepSeek-V3 Technical Report*](https://arxiv.org/abs/2412.19437)

## 2. Modules
- `.modules`: building blocks
    - `.activation`: Activation functions.
    - `.attention`: Attention functions and modules.
    - `.ffn`: FFN and MoE modules.
    - `.positional_encoding`: Functions for generating positional encoding.
- `.rl`: reinforcement learning of LLMs & LLM Agent
- `.deepseek`: DeepSeek family
- `.classic`: Classic models like BERT, Transformer.
- `.linear`: Model with linear attention