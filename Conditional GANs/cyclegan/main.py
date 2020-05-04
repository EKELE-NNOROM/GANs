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

from models import GeneratorResNet, Discriminator, weights_init_normal
from datasets import Monet2PhotoDataset
from utils import ReplayBuffer, LambdaLR

import torch.nn as nn
import torch.nn.functional as F
import torch

from device import cuda, Tensor
from device import criterion_GAN, criterion_cycle, criterion_identity
from PIL import Image

#import models

def get_parser():
    from argparse import ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--decay_epoch", type=int, default=10, help="dimensionality of the latent space")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval betwen image samples")
    parser.add_argument("--checkpoint_interval", type=int, default=2, help="interval between saving model checkpoints")
    parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks")
    parser.add_argument("--data_root", type=str, default="monet2photo", help="path to monet2photo dataset")
    parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
    parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
    parser.add_argument("--img_height", type=int, default=256, help="image height")
    parser.add_argument("--img_width", type=int, default=256, help="image height")
    return parser


def sample_image(args, val_dataloader, G_AB, G_BA, Tensor, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    imgs = next(iter(val_dataloader))
    G_AB.eval()
    G_BA.eval()
    real_A = Variable(imgs["A"].type(Tensor))
    fake_B = G_AB(real_A)
    real_B = Variable(imgs["B"].type(Tensor))
    fake_A = G_BA(real_B)
    # Arrange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    # Arrange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, "images/{}/{}.png".format(args.data_root, batches_done), normalize=False)

# def losses(cuda):
#     criterion_GAN = torch.nn.MSELoss()
#     criterion_cycle = torch.nn.L1Loss()
#     criterion_identity = torch.nn.L1Loss()
#     return criterion_GAN.cuda(), criterion_cycle.cuda(), criterion_identity.cuda()

def main(args):

    # Create sample and checkpoint directories
    os.makedirs("images/{}".format(args.data_root), exist_ok=True)
    os.makedirs("saved_models/{}".format(args.data_root), exist_ok=True)

    input_shape = (args.channels, args.img_height, args.img_width)

    # Initialize generator and discriminator
    G_AB = GeneratorResNet(input_shape, args.n_residual_blocks)
    G_BA = GeneratorResNet(input_shape, args.n_residual_blocks)
    D_A = Discriminator(input_shape)
    D_B = Discriminator(input_shape)

    if cuda:
        G_AB = G_AB.cuda()
        G_BA = G_BA.cuda()
        D_A = D_A.cuda()
        D_B = D_B.cuda()

    if args.epoch != 0:
    # Load pretrained models
        G_AB.load_state_dict(torch.load("saved_models/{}/G_AB_{}.pth" % (args.data_root, args.epoch)))
        G_BA.load_state_dict(torch.load("saved_models/{}/G_BA_{}.pth" % (args.data_root, args.epoch)))
        D_A.load_state_dict(torch.load("saved_models/{}/D_A_{}.pth" % (args.data_root, args.epoch)))
        D_B.load_state_dict(torch.load("saved_models/{}/D_B_{}.pth" % (args.data_root, args.epoch)))
    else:
        # Initialize weights
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)


    # Optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # Learning rate update schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    #os.listdir(data_root)
    # img_shape = (args.channels, args.img_size, args.img_size)
    #
    # cuda = True if torch.cuda.is_available() else False
    #
    # # Loss function
    # adversarial_loss = torch.nn.BCELoss()
    # auxiliary_loss = torch.nn.CrossEntropyLoss()
    #
    # # Initialize generator and discriminator
    # generator, discriminator = models.main(args)
    #
    # if cuda:
    #     generator.cuda()
    #     discriminator.cuda()
    #     adversarial_loss.cuda()
    #     auxiliary_loss.cuda()
    #
    # generator.apply(weights_init_normal)
    # discriminator.apply(weights_init_normal)
    # print('Generator: ', generator)
    # print('Discriminator: ', discriminator)

    # dataset = datasets.ImageFolder(root=args.data_root,
    #                        transform=transforms.Compose([
    #                            transforms.Resize(args.img_size),
    #                            transforms.CenterCrop(args.img_size),
    #                            transforms.ToTensor(),
    #                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                        ]))
    # # Create the dataloader
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
    #                                          shuffle=True, num_workers=2)


    transforms_ = [
    transforms.Resize(int(args.img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((args.img_height, args.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    # Training data loader
    dataloader = DataLoader(
        Monet2PhotoDataset("../../datasets/%s"%args.data_root, transforms_=transforms_, unaligned=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpu,
    )
    # Test data loader
    val_dataloader = DataLoader(
        Monet2PhotoDataset("../../datasets/%s"%args.data_root, transforms_=transforms_, unaligned=True, mode="test"),
        batch_size=5,
        shuffle=True,
        num_workers=1,
    )

    optimizer_list = [optimizer_G, optimizer_D_A, optimizer_D_B]
    network_list = [G_AB, G_BA, D_A, D_B]
    buffer_list = [fake_A_buffer, fake_B_buffer]
    scheduler_list = [lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B]
    dataloaders = [dataloader, val_dataloader]
    train(args, network_list, buffer_list, optimizer_list, scheduler_list, dataloaders)

def train(args, network_list, buffer_list, optimizer_list, scheduler_list, dataloaders):
    optimizer_G, optimizer_D_A, optimizer_D_B = optimizer_list[0], optimizer_list[1], optimizer_list[2]
    G_AB, G_BA, D_A, D_B = network_list[0], network_list[1], network_list[2], network_list[3]
    fake_A_buffer, fake_B_buffer = buffer_list[0], buffer_list[1]
    lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B = scheduler_list[0], scheduler_list[1], scheduler_list[2]
    dataloader, val_dataloader = dataloaders[0], dataloaders[1]
    prev_time = time.time()
    for epoch in range(0, args.n_epochs):
        for i, batch in enumerate(dataloader):
            # Set model input
            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            G_AB.train()
            G_BA.train()

            optimizer_G.zero_grad()

            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)

            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss
            loss_G = loss_GAN + args.lambda_cyc * loss_cycle + args.lambda_id * loss_identity

            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------

            optimizer_D_A.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2

            loss_D_A.backward()
            optimizer_D_A.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------

            optimizer_D_B.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_B(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2

            loss_D_B.backward()
            optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2

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
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                % (
                    epoch,
                    args.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_cycle.item(),
                    loss_identity.item(),
                    time_left,
                )
            )

            # If at sample interval save image
            if batches_done % args.sample_interval == 0:
                sample_image(args, val_dataloader, G_AB, G_BA, Tensor, batches_done)

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()
        args.epoch = epoch
        print('helper current epoch ',args.epoch)

        if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(G_AB.state_dict(), "saved_models/{}/G_AB_{}.pth".format(args.data_root, epoch))
            torch.save(G_BA.state_dict(), "saved_models/{}/G_BA_{}.pth".format(args.data_root, epoch))
            torch.save(D_A.state_dict(), "saved_models/{}/D_A_{}.pth".format(args.data_root, epoch))
            torch.save(D_B.state_dict(), "saved_models/{}/D_B_{}.pth".format(args.data_root, epoch))
























#     train(generator, discriminator, dataloader, args, cuda, adversarial_loss, auxiliary_loss)
#
# def train(generator, discriminator, dataloader, args, cuda, adversarial_loss, auxiliary_loss):
#
#     optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
#     optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
#
#     for epoch in range(args.n_epochs):
#         for i, (imgs, labels) in enumerate(dataloader):
#
#             batch_size = imgs.shape[0]
#
#             # Adversarial ground truths
#             valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
#             fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
#
#             # Configure input
#             real_imgs = Variable(imgs.type(FloatTensor))
#             labels = Variable(labels.type(LongTensor))
#
#             # -----------------
#             #  Train Generator
#             # -----------------
#
#             optimizer_G.zero_grad()
#
#             # Sample noise as generator input
#             z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, args.latent_dim))))
#             gen_labels = Variable(LongTensor(np.random.randint(0, args.n_classes, batch_size)))
#
#             # Generate a batch of images
#             gen_imgs = generator(z, gen_labels)
#
#             # Loss measures generator's ability to fool the discriminator
#             validity, pred_label = discriminator(gen_imgs)
#             g_loss = 0.5 * adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels)
#
#             g_loss.backward()
#             optimizer_G.step()
#
#             # ---------------------
#             #  Train Discriminator
#             # ---------------------
#
#             optimizer_D.zero_grad()
#
#             # Loss for real images
#             real_pred, real_aux = discriminator(real_imgs)
#             d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2
#
#             # Loss for fake images
#             fake_pred, fake_aux = discriminator(gen_imgs.detach())
#             d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2
#
#             # Measure discriminator's ability to classify real from generated samples
#             d_loss = (d_real_loss + d_fake_loss) / 2
#
#             # Calculate discriminator accuracy
#             pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
#             gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
#             d_acc = np.mean(np.argmax(pred, axis=1) == gt)
#
#             d_loss.backward()
#             optimizer_D.step()
#
#             print(
#                 "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
#                 % (epoch, args.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
#             )
#
#             batches_done = epoch * len(dataloader) + i
#             if batches_done % args.sample_interval == 0:
#                 sample_image(args, generator, n_row=10, batches_done=batches_done)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    #os.makedirs("images", exist_ok=True)
    main(args)
