import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)  # 65 in this case (26 + 26 + miscellaneous char(s))

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

# All encoded data from the `input.txt`
data = torch.tensor(encode(text), dtype=torch.long)

# Let's now split up the data into train and validation sets
n = int(0.9 * len(data))  # first 90% will be train, rest val(for evaluation)
train_data = data[:n]
val_data = data[n:]


torch.manual_seed(1337)
batch_size = 4  # how many independent sequences will we process in parallel?
block_size = 8  # context length, what is the maximum context length for predictions


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


xb, yb = get_batch("train")


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B, T, C)

        B, T, C = logits.shape
        logits = logits.view(B * T, C)  # PyTorch expects a (B, C, T) rather
        loss = F.cross_entropy(logits, targets)

        return logits, loss


m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits, loss)
