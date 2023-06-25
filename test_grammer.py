import torch
import torch.nn as nn
import torch.nn.functional as F

emb = torch.randn(2,25,8,8,5,5)
y = emb.sum(dim=[4,5])
print(y.shape)