import urllib.request
import os
import tempfile

def get_or_download_file(file_path):
  if os.path.exists(file_path):
    return file_path

  url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

  temp_dir = tempfile.gettempdir()
  temp_file_path = os.path.join(temp_dir, os.path.basename(file_path))
  if not os.path.exists(temp_file_path):
    urllib.request.urlretrieve(url, temp_file_path)

  return temp_file_path

def initialize_tokenizer():
  file = get_or_download_file("/tmp/input.txt")

  ctoi = {}
  itoc = {}

  with open(file, "r", encoding="utf-8") as f:
    content = f.read()
    vocab = list(set(content))
    vocab.sort()

    ctoi = {ch: i for i, ch in enumerate(vocab)}
    itoc = {i: ch for i, ch in enumerate(vocab)}

    output = []
    for c in content:
      output.append(ctoi[c])



def main():
  encoder = initialize_tokenizer()

if __name__ == "__main__":
  main()