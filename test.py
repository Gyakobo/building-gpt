import torch

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

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
