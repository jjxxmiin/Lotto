import torch
from torch.utils.data import Dataset
from utils import one_hot_encoding

class LottoDataset(Dataset):
    def __init__(self, data, view=1):
        super().__init__()

        self.data = data
        self.view = view

    def __len__(self):
        return len(self.data) - self.view
    
    def __getitem__(self, idx):
        X = []
        for d in self.data[idx:idx+self.view]:
            X.append(one_hot_encoding(d))
            
        y = one_hot_encoding(self.data[idx+self.view])
        
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y).unsqueeze_(0)
        
        return X, y