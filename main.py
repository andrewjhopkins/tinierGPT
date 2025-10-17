from tokenizer import Tokenizer
from model import Model
import torch

batch_size = 10
block_size = 32
embedding_size = 64
n_head = 4  # Changed from 3 to 4 so it divides evenly into 64
n_layer = 3

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device used: {device}")

    file_path = "/tmp/input.txt"
    tokenizer = Tokenizer(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    train_tokens = tokenizer.encode(content)

    # 10% for validation
    n = int(0.9 * len(train_tokens))

    train_tokens = torch.tensor(train_tokens[:n], dtype=torch.long, device=device)
    model = Model(tokenizer.vocab_size, embedding_size, block_size, n_head, n_layer).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    iters = 100

    for iter in range(iters):
        xb, yb = get_batch(train_tokens, batch_size, block_size, device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print("loss: ", loss.item())
    
def get_batch(tokens, batch_size, block_size, device):
    ix = torch.randint(0, len(tokens) - block_size, (batch_size,), device=device)
    x = torch.stack([tokens[i:i+block_size] for i in ix])
    y = torch.stack([tokens[i+1:i+block_size+1] for i in ix])
    return x, y
      
if __name__ == "__main__":
    main()