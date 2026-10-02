# Building a GPT

![image](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![image](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![image](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![image](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![image](https://img.shields.io/badge/windows%20terminal-4D4D4D?style=for-the-badge&logo=windows%20terminal&logoColor=white)

Author: [Andrew Gyakobo](https://github.com/Gyakobo)

A special thanks to [Andrej Karpathy](https://github.com/karpathy) and his lecture [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY), which guided me through every line of this project.

>[!NOTE]
>This is my first transformer built from scratch. It's a learning project and a direct follow-up to my [Makemore-from-scratch](https://github.com/Gyakobo/Makemore-from-scratch) work.

This project implements a small, decoder-only **Generative Pre-trained Transformer (GPT)** in PyTorch. It is trained on the complete works of Shakespeare and learns to generate new Shakespeare-like text one character at a time.

## Introduction

>**GPT** is a decoder-only transformer: a stack of self-attention and feed-forward blocks trained to predict the next token in a sequence, given every token that came before it. The architecture comes from *Attention Is All You Need* [Vaswani et al., 2017].

The model in this repository is the same idea at a much smaller scale. Instead of words or sub-word tokens, it works directly on **characters**, so the whole vocabulary is just 65 symbols (upper and lowercase letters, punctuation, spaces and newlines).

Just a few notes about the model:
* ~10.8 million trainable parameters
* 6 transformer blocks, 6 attention heads each, 384-dimensional embeddings
* A context window of 256 characters
* Trained on ~1.1 MB of text ([input.txt](./input.txt), the *Tiny Shakespeare* dataset)

## Methodology

### 1) Tokenization

Every unique character in `input.txt` is sorted and mapped to an integer. Two lookup tables (`stoi` and `itos`) then encode text into integers and decode integers back into text.

```python
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])
```

### 2) Train / validation split

The encoded text becomes one long tensor. The first 90% is used for training and the last 10% for validation, so we can check whether the model is actually learning or just memorizing.

### 3) Batching

`get_batch()` samples 64 random chunks of 256 characters. The targets `y` are simply the inputs `x` shifted one character to the right, so every position in a chunk is its own training example.

```
x: F i r s t   C i t i z e
y: i r s t   C i t i z e n
```

### 4) The model

Each character goes through the following pipeline:

```
character indices (B, T)
        │
token embedding + position embedding   (B, T, 384)
        │
┌───────────────────────────────┐
│  LayerNorm → Multi-Head Attn  │ + residual
│  LayerNorm → Feed-Forward     │ + residual   × 6 blocks
└───────────────────────────────┘
        │
final LayerNorm
        │
linear head → logits over 65 characters (B, T, 65)
```

>[!NOTE]
>`B` is the batch size, `T` is the time dimension (context length) and `C` is the channel (embedding) dimension.

## Code snippets

* **`Head`** — one head of self-attention. Every token emits a *query* ("what am I looking for?") and a *key* ("what do I contain?"). Their dot product gives the attention scores. A lower-triangular mask stops tokens from looking into the future, which is what makes this a **decoder** block.

```python
wei = q @ k.transpose(-2, -1) * C**-0.5          # (B, T, T) affinities
wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)
out = wei @ v                                     # weighted sum of values
```

* **`MultiHeadAttention`** — runs 6 heads in parallel, concatenates their outputs and projects them back to the embedding size.

* **`FeedFoward`** — a two-layer MLP (`384 → 1536 → 384`) with a ReLU in between. Attention is the *communication* step between tokens, and this is the *computation* step each token does on its own.

* **`Block`** — combines the two, with pre-LayerNorm and residual connections so gradients flow cleanly through a deep stack.

```python
def forward(self, x):
    x = x + self.sa(self.ln1(x))     # communication
    x = x + self.ffwd(self.ln2(x))   # computation
    return x
```

* **`BigramLanguageModel`** — the full GPT. The name is left over from the bigram baseline it started as.

* **`generate()`** — crops the context to the last 256 characters, takes the logits of the final position, samples the next character from the softmax distribution and appends it. Repeat.

## Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `batch_size` | 64 | Sequences processed in parallel |
| `block_size` | 256 | Maximum context length |
| `max_iters` | 5000 | Training steps |
| `eval_interval` | 500 | Steps between loss reports |
| `eval_iters` | 200 | Batches averaged per loss estimate |
| `n_embd` | 384 | Embedding dimension |
| `n_head` | 6 | Attention heads per block |
| `n_layer` | 6 | Transformer blocks |
| `dropout` | 0.2 | Dropout rate |
| optimizer | AdamW | `lr = 1e-3` |

## Getting started

```bash
git clone https://github.com/Gyakobo/building-gpt.git
cd building-gpt
pip install -r requirements.txt
python bigram.py
```

>[!IMPORTANT]
>A CUDA-capable GPU is strongly recommended. The script picks `cuda` automatically when available; on a CPU, 5000 iterations of a 10.8M-parameter model will take a very long time. To try it on a CPU, lower `n_layer`, `n_embd` and `block_size`.

The script prints the training and validation loss every 500 steps and then generates 500 characters of new text.

## Results

<!-- Paste your own loss log and a generated sample below -->

```
step 0: train loss ..., val loss ...
...
step 4500: train loss ..., val loss ...
```

Sample output:

```
(paste generated text here)
```

>[!NOTE]
>For reference, Karpathy's equivalent model reaches a validation loss of about **1.48**. The output reads like Shakespeare at a glance (character names, verse line breaks, archaic words) but doesn't make sense on closer reading. That's expected from a character-level model of this size.

## Future work

* Finish `v2.py` as a cleaned-up version of the model
* Switch from character-level tokens to a sub-word tokenizer (BPE)
* Add model checkpointing so training doesn't have to restart from scratch
* Plot the training and validation loss curves

## Reference

* Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.

* He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 770-778.

* Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization. *arXiv preprint arXiv:1607.06450*.

* Karpathy, A. (2023). *Let's build GPT: from scratch, in code, spelled out.* [YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY) / [nanoGPT](https://github.com/karpathy/nanoGPT).

## License
MIT