import torch
import torch.nn as nn

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

        # TODO:
        self.blocks = nn.Sequential(*[Block(embedding_size, n_head) for _ in range(n_layer)])

    def train(input):
        return

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        # layer norm
        self.ln1 = nn.LayerNorm(n_embd)
        # self attention
        self.sa = MultiHeadAttention(n_head, head_size)
        # layer norm
        self.ln2 = nn.LayerNorm(n_embd)
        # feed forward
        self.ffwd = FeedForward(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Decoder:
    def forward():
        # masked multi head attention
        # layer norm
        # feed forward
        # layer norm


