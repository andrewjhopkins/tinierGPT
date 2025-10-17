import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 10
block_size = 32
embedding_size = 64
vocab_size = 1234 # change this to tokenizer vocab size
n_head = 3
n_layer = 3

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding_table = nn.Embedding(block_size, embedding_size)
        self.blocks = nn.Sequential(*[Block(embedding_size, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(embedding_size)
        self.lm_head = nn.Linear(embedding_size, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        batch, time = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(time, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            batch, time, channels = logits.shape
            logits = logits.view(batch * time, channels)
            targets = targets.view(batch * time)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd // n_head
        # layer norm
        self.ln1 = nn.LayerNorm(n_embd)
        # self attention
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size)
        # layer norm
        self.ln2 = nn.LayerNorm(n_embd)
        # feed forward
        self.ffwd = FeedForward(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.ReLU(),
            nn.Linear(4 * embedding_size, embedding_size)
        )
    
    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()
        self.head = nn.ModuleList([Head(head_size, n_embd, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        self.keys = nn.Linear(n_embd, head_size, bias=False)
        self.queries = nn.Linear(n_embd, head_size, bias=False)
        self.values = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        batch, time, channels = x.shape
        k = self.keys(x)
        q = self.queries(x)
        wei = q @ k.transpose(-2, -1) * channels ** -0.5

        wei = wei.masked_fill(self.trill[:time, :time] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        v = self.values(x)
        out = wei @ v
        return out