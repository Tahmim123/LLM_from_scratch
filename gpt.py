import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

block_size = 8
vocab_size = 65
batch_size = 4
eval_iters = 200
eval_interval = 300
learning_rate = 1e-2
max_iters = 3000

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda st: [stoi[char] for char in st]
decode = lambda st: ''.join(itos[char] for char in st)

#train and test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    input_tokens = torch.stack([data[i: i + block_size] for i in ix])
    target_tokens = torch.stack([data[i+1: i + block_size + 1] for i in ix])
    # print(input_tokens)
    return input_tokens, target_tokens    

@torch.no_grad()
def estimate_loss():
    out = {}
    model = GptLanguageModel(vocab_size)
    for split in ['train', 'val_data']:
        loss_per_iter = torch.zeros(eval_iters)
        for step in eval_iters:
            input, target = get_batch(split)
            logits, loss = model(input, target)
            loss_per_iter[step] = loss
        out[split] = loss_per_iter.mean()
    model.train()
    return out

class GptLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # if we have (65,65) embedding lookup table, each logit represents one of the characters
        self.embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.embedding_table(idx)
        if targets == None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            # print(logits.shape)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            logits, loss = self(idx) 
            logits = logits[:, -1, :] # this is taking the last row from the time dimension (B*C) (4,8,65) -> (4,65)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

model = GptLanguageModel(vocab_size)
m = model.to(device)
input, output =  get_batch('train')

optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        out = estimate_loss()
        print(out)

    input, output = get_batch('train')
    logits, loss = model(input, output)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    
