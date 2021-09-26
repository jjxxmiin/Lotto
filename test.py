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

view = 4
use_column = ['Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19']
device = 'cuda'
load_path = 'checkpoint/lotto_4_2.pth'

df = pd.read_excel('data/lotto.xlsx')
df = df[use_column][2:]
df = df.astype('int')

total_dataset = LottoDataset(df.values, view=view)
_, test_dataset = random_split(total_dataset, [int(len(total_dataset) * 0.95), len(total_dataset) - int(len(total_dataset) * 0.95)])

test_loader = DataLoader(test_dataset, 
                          batch_size=1, 
                          shuffle=False)

model = LSTM().to(device)
model.load_state_dict(torch.load(load_path))

model.eval()
for X, y in test_loader:
    X = X.to(device)
    y = y.to(device)
    
    with torch.no_grad():
        pred = model(X)
        
    print("===============================")
    values, indexs = torch.topk(pred, k=7)
    print(indexs + 1)
    values, indexs = torch.topk(y, k=7)
    print(indexs + 1)
    print("===============================")