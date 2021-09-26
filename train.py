import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

from dataset import LottoDataset
from model import LSTM
from utils import set_randomness

random_seed = 777

set_randomness(random_seed)

view = 36
batch_size = 2
use_column = ['Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19']
device = 'cuda'
lr = 0.001
epochs = 100
save_path = f'checkpoint/lotto_{view}_{batch_size}.pth'

df = pd.read_excel('data/lotto.xlsx')
df = df[use_column][2:]
df = df.astype('int')

total_dataset = LottoDataset(df.values, view=view)
train_dataset, test_dataset = random_split(total_dataset, [int(len(total_dataset) * 0.95), len(total_dataset) - int(len(total_dataset) * 0.95)])
train_dataset, valid_dataset = random_split(train_dataset, [int(len(train_dataset) * 0.95), len(train_dataset) - int(len(train_dataset) * 0.95)])

train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)

valid_loader = DataLoader(valid_dataset, 
                          batch_size=batch_size, 
                          shuffle=False)

model = LSTM().to(device)

criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

train_total = len(train_loader)
valid_total = len(valid_loader)

min_loss = 100

for e in range(epochs):
    train_loss = 0
    valid_loss = 0
    
    model.train()
    for X, y in tqdm(train_loader, total=train_total):
        optimizer.zero_grad()
        
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = criterion(pred, y)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    model.eval()
    for X, y in tqdm(valid_loader, total=valid_total):
        X = X.to(device)
        y = y.to(device)
        
        with torch.no_grad():
            pred = model(X)
            loss = criterion(pred, y)
        
        valid_loss += loss.item()
        
    train_loss = train_loss / train_total
    valid_loss = valid_loss / valid_total
    
    print(f"Train Loss / {train_loss}")
    print(f"Valid Loss / {valid_loss}")
    
    torch.save(model.state_dict(), save_path)