from tokenizer import Tokenizer
from model import Model

def main():
    file_path = "/tmp/input.txt"
    tokenizer = Tokenizer("/tmp/input.txt")
    model = Model()

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    data = tokenizer.encode(content)
    print(data)
      
if __name__ == "__main__":
    main()