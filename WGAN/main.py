import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch


from models import Generator, Discriminator
from device import cuda, Tensor

def get_parser():
    from argparse import ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--lambda_gp", type=float, default=10)
    return parser


def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    logits = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    gradients = autograd.grad(
                              outputs=logits,
                              inputs=interpolates,
                              grad_outputs=fake,
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def main(args):

    img_shape = (args.channels, args.img_size, args.img_size)

    cuda = True if torch.cuda.is_available() else False

    # Loss function
    #lambda_gp = 10
    #adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator(img_shape)
    discriminator = Discriminator()

    if cuda:
        generator.cuda()
        discriminator.cuda()

    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(args.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    train(generator, discriminator, dataloader, args)

def train(generator, discriminator, dataloader, args):

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    batches_done = 0
    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):


            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))))

            gen_imgs = generator(z)

            # Real images
            real_validity = discriminator(real_imgs)

            # Fake images
            fake_validity = discriminator(gen_imgs)

            # Gradient Penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, gen_imgs.data)

            # adversarial_loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + args.lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % args.n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------

                gen_imgs = generator(z)
                fake_validity = discriminator(gen_imgs)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()


                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, args.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

                if batches_done % args.sample_interval == 0:
                    save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

                batches_done += args.n_critic


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    os.makedirs("images", exist_ok=True)
    main(args)
