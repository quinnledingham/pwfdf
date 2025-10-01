import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from app import App, Data, App_Dataset
from data import PWFDF_Data
from eval import evaluate

params = {
  "epochs": 20,
  "batch_size": 64,
  "max_iters": 5000,
  "eval_interval": 500,
  "learning_rate": 3e-4,
  "eval_iters": 200,
  "model_output": './output/transformer.model',
  "optimizer": 'AdamW',
  "dataset_name": 'Custom',
}


block_size = 256
n_embd = 128
n_head = 4
n_layer = 6
dropout = 0.2
head_size = n_embd // n_head

class Head(nn.Module):
  """ one head of self-attention """

  def __init__(self):
      super().__init__()
      self.key = nn.Linear(n_embd, head_size, bias=False)
      self.query = nn.Linear(n_embd, head_size, bias=False)
      self.value = nn.Linear(n_embd, head_size, bias=False)
      self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

      self.dropout = nn.Dropout(dropout)

  def forward(self, x):
      # input of size (batch, time-step, channels)
      # output of size (batch, time-step, head size)
      B,T,C = x.shape
      k = self.key(x)   # (B,T,hs)
      q = self.query(x) # (B,T,hs)
      # compute attention scores ("affinities")
      wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
      #wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
      wei = F.softmax(wei, dim=-1) # (B, T, T)
      wei = self.dropout(wei)
      # perform the weighted aggregation of the values
      v = self.value(x) # (B,T,hs)
      out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
      return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.sa = MultiHeadAttention()
        self.ffwd = FeedFoward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Transformer(nn.Module):
  def __init__(self, input_size):
    super().__init__()
    # each token directly reads off the logits for the next token from a lookup table
    #self.token_embedding_table = nn.Embedding(app.vocab_size, app.n_embd)
    #self.position_embedding_table = nn.Embedding(app.block_size, app.n_embd)

    self.input_projection = torch.nn.Linear(input_size, n_embd)
    self.pos_encoding = torch.nn.Parameter(torch.randn(1, 1, n_embd))

    self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd) # final layer norm
    self.lm_head = nn.Linear(n_embd, input_size)

    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
          torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, idx, targets=None):
    B, T = idx.shape

    # idx and targets are both (B,T) tensor of integers
    #tok_emb = self.token_embedding_table(idx) # (B,T,C)
    #x = tok_emb + pos_emb # (B,T,C)

    x = idx.unsqueeze(1)
    x = self.input_projection(x)
    x = x + self.pos_encoding

    x = self.blocks(x) # (B,T,C)
    x = self.ln_f(x) # (B,T,C)
    logits = self.lm_head(x) # (B,T,vocab_size)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

def train(app):
  print("Training")

  for epoch in range(app.epochs):
    app.model.train()
    pbar = app.pbar(epoch)

    for batch_i, (features, labels) in enumerate(pbar):
      outputs, loss = app.model(features, labels)
      app.optimizer.zero_grad()
      loss.backward()
      app.optimizer.step()

    print(f"\nEpoch {epoch+1}/{app.epochs}")
    train_metrics = evaluate(app.model, app.data.train_loader)
    print("Train")
    train_metrics.output()
    test_metrics = evaluate(app.model, app.data.test_loader)
    print("Test")
    test_metrics.output()

def test(app):
  app.model.eval()
  test_loss = 0
  test_correct = 0
  test_total = 0

  test_pbar = app.test_pbar()

  with torch.no_grad():
    for i, (data, target) in enumerate(test_pbar):
      output, loss = app.model(data, target)

      test_loss += loss.item()
      _, predicted = output.max(1)
      test_total += target.size(0)
      test_correct += predicted.eq(target).sum().item()

  test_loss = test_loss / len(app.data.test_loader)
  test_acc = 100.0 * test_correct / test_total

  print(f"Final Test Accuracy: {test_acc:.2f}%")

app = App(params)

def main():
  # loading data
  pwfdf_data = PWFDF_Data()

  data, target = pwfdf_data.get_data_target()
  data = torch.FloatTensor(data)
  target = torch.tensor(target)

  X_train, X_val, y_train, y_val = train_test_split(data, target, test_size=0.1, random_state=42, stratify=target)

  train_dataset = app.create_dataset(X_train, y_train)
  test_dataset = app.create_dataset(X_val, y_val)

  app.data = Data(app.batch_size, train_dataset, test_dataset)

  app.model = Transformer(data.shape[1])
  app.set_optimizer()
  app.train_func = train
  app.test_func = test
  app.main()


if __name__ == "__main__":
  main()
