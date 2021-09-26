import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        
        self.lstm = nn.LSTM(45, 
                            128, 
                            num_layers=3, 
                            dropout=dropout, 
                            batch_first=True)
        
        self.linear = nn.Linear(128, 45)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x[:,-1:,:])
        return x
    
    
if __name__ == "__main__":
    sample = torch.rand(1, 7, 45)
    
    model = LSTM()
    
    output = model(sample)
    
    print(output.shape)