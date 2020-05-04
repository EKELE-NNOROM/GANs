import torch

cuda = True if torch.cuda.is_available() else False

float_tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
long_tensor = torch.cuda.LongTensor if cuda else torch.LongTensor
