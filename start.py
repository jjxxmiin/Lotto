import pandas as pd
import torch
from model import LSTM
from utils import one_hot_encoding

views = [4, 12, 24, 36, 48]
use_column = ['Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19']
device = 'cuda'

df = pd.read_excel('data/lotto.xlsx')
df = df[use_column][2:]
df = df.astype('int')

for i, view in enumerate(range(views)):
    X = []
    for d in df.values[-view:]:
        X.append(one_hot_encoding(d))
    X = torch.FloatTensor(X).unsqueeze(0).to(device)
    
    model = LSTM().to(device)
    model.load_state_dict(torch.load(f'checkpoint/lotto_{view}_2.pth'))
    model.eval()

    with torch.no_grad():
        pred = model(X)
        
    print("===============================")
    values, indexs = torch.topk(pred, k=7)
    print(f"이번 주 로또 번호 {i} : {(indexs + 1).detach().cpu().numpy()[0][0]}")
    print("===============================")