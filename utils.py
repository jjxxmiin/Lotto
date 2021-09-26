import torch
import numpy as np
import random


def set_randomness(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    
def one_hot_encoding(data):
    encoding = np.zeros(45)

    for d in data:
        encoding[d - 1] = 1
        
    return encoding