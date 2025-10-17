import urllib.request
import os
import tempfile

class Tokenizer():
  def __init__(self, file_path):
    file_path = get_or_download_file(file_path)
    self.file_path = file_path
    self.encoder = {}
    self.decoder = {}
    self.initialize_encoder_decoder()
  
  def encode(self, text):
    return [self.encoder[ch] for ch in text]

  def decode(self, data):
    return ''.join([self.decoder[i] for i in data])


  def initialize_encoder_decoder(self):
    file = get_or_download_file(self.file_path)

    with open(file, "r", encoding="utf-8") as f:
      content = f.read()
      vocab = list(set(content))
      vocab.sort()

      self.encoder = {ch: i for i, ch in enumerate(vocab)}
      self.decoder = {i: ch for i, ch in enumerate(vocab)}

def get_or_download_file(file_path):
  if os.path.exists(file_path):
    return file_path

  url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

  temp_dir = tempfile.gettempdir()
  temp_file_path = os.path.join(temp_dir, os.path.basename(file_path))
  if not os.path.exists(temp_file_path):
    urllib.request.urlretrieve(url, temp_file_path)

  return temp_file_path