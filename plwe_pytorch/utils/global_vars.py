import torch

global dtype
# Using torch.FloatTensor if you use CPU
# dtype = torch.FloatTensor
# Using torch.cuda.FloatTensor if you use GPU
dtype = torch.cuda.FloatTensor
