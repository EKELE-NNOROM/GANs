import torch

cuda = torch.cuda.is_available()

criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()



if cuda:
    Tensor = torch.cuda.FloatTensor
    criterion_GAN = criterion_GAN.cuda()
    criterion_pixelwise = criterion_pixelwise.cuda()
