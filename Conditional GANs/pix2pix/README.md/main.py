import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import GeneratorUNet, Discriminator, weights_init_normal
from datasets import FacadeDataset


import torch.nn as nn
import torch.nn.functional as F
import torch

from device import cuda, Tensor
from device import criterion_GAN, criterion_pixelwise
from PIL import Image

#import models

def get_parser():
    from argparse import ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--decay_epoch", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval betwen image samples")
    parser.add_argument("--checkpoint_interval", type=int, default=2, help="interval between saving model checkpoints")

    parser.add_argument("--data_root", type=str, default="facades", help="path to monet2photo dataset")

    parser.add_argument("--img_height", type=int, default=256, help="image height")
    parser.add_argument("--img_width", type=int, default=256, help="image height")
    parser.add_argument("--lambda_pixel", type=int, default=100, help="Hyperparameter for lambda pixel")
    return parser


def sample_image(args, val_dataloader, generator, Tensor, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    imgs = next(iter(val_dataloader))

    real_A = Variable(imgs["B"].type(Tensor))
    fake_B = generator(real_A)
    real_B = Variable(imgs["A"].type(Tensor))

    # Arrange images along y-axis
    image_grid = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    save_image(image_grid, "images/{}/{}.png".format(args.data_root, batches_done), nrow=5, normalize=True)


def main(args):

    # Create sample and checkpoint directories
    os.makedirs("images/{}".format(args.data_root), exist_ok=True)
    os.makedirs("saved_models/{}".format(args.data_root), exist_ok=True)

    # Loss weight of L1 pixel-wise loss between translated image and real image
    lambda_pixel = 100

    # Initialize generator and discriminator
    generator = GeneratorUNet()
    discriminator = Discriminator()

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    if args.epoch != 0:
    # Load pretrained models
        generator.load_state_dict(torch.load("saved_models/{}/generator_{}.pth" % (args.data_root, args.epoch)))
        discriminator.load_state_dict(torch.load("saved_models/{}/discriminator_{}.pth" % (args.data_root, args.epoch)))
    else:
        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)


    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    transforms_ = [
    transforms.Resize((args.img_height, args.img_width), Image.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    # Training data loader
    dataloader = DataLoader(
        FacadeDataset("../../datasets/{}".format(args.data_root), transforms_=transforms_),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpu,
    )
    # Test data loader
    # global val_dataloader
    val_dataloader = DataLoader(
        FacadeDataset("../../datasets/{}".format(args.data_root), transforms_=transforms_,  mode="val"),
        batch_size=10,
        shuffle=True,
        num_workers=1,
    )

    optimizer_list = [optimizer_G, optimizer_D]
    network_list = [generator, discriminator]

    dataloaders = [dataloader, val_dataloader]
    train(args, network_list, optimizer_list, dataloaders)

def train(args, network_list, optimizer_list, dataloaders):
    optimizer_G, optimizer_D = optimizer_list[0], optimizer_list[1]
    generator, discriminator = network_list[0], network_list[1]
    dataloader, val_dataloader = dataloaders[0], dataloaders[1]

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, args.img_height // 2 ** 4, args.img_width // 2 ** 4)

    prev_time = time.time()
    for epoch in range(0, args.n_epochs):
        for i, batch in enumerate(dataloader):
            # Set model input
            real_A = Variable(batch["B"].type(Tensor))
            real_B = Variable(batch["A"].type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # GAN loss
            fake_B = generator(real_A)
            pred_fake = discriminator(fake_B, real_A)
            loss_GAN = criterion_GAN(pred_fake, valid)

            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_B, real_B)

            # Total loss
            loss_G = loss_GAN + args.lambda_pixel * loss_pixel

            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------

            optimizer_D.zero_grad()

            # Real loss
            pred_real = discriminator(real_B, real_A)
            loss_real = criterion_GAN(pred_real, valid)
            # Fake loss (on batch of previously generated samples)
            pred_fake = discriminator(fake_B.detach(), real_A)
            loss_fake = criterion_GAN(pred_fake, fake)
            # Total loss
            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = args.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
                % (
                    epoch,
                    args.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_pixel.item(),
                    loss_GAN.item(),
                    time_left,
                )
            )

            # If at sample interval save image
            if batches_done % args.sample_interval == 0:
                sample_image(args, val_dataloader, generator, Tensor, batches_done)

        if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models/{}/generator_{}.pth".format(args.data_root, epoch))
            torch.save(discriminator.state_dict(), "saved_models/{}/discriminator_{}.pth".format(args.data_root, epoch))

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
