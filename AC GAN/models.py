import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import argparse


def get_parser():
    from argparse import ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
    return parser

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(args.n_classes, args.latent_dim)
        self.init_size = args.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(args.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
                                         nn.BatchNorm2d(128),
                                         nn.Upsample(scale_factor=2),
                                         nn.Conv2d(128, 128, 3, stride=1, padding=1),
                                         nn.BatchNorm2d(128, 0.8),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Upsample(scale_factor=2),
                                         nn.Conv2d(128, 64, 3, stride=1, padding=1),
                                         nn.BatchNorm2d(64, 0.8),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Conv2d(64, args.channels, 3, stride=1, padding=1),
                                         nn.Tanh(),

        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
                                   *discriminator_block(args.channels, 16, bn=False),
                                   *discriminator_block(16, 32),
                                   *discriminator_block(32, 64),
                                   *discriminator_block(64, 128),
        )

        ds_size = args.img_size // 2 ** 4

        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, args.n_classes), nn.Softmax())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label


def main(args):
    generator = Generator(args)
    discriminator = Discriminator(args)
    return generator, discriminator

if __name__ == "__main__":
    parser = get_parser()
    print(parser)
    args = get_parser.parse_args()
    print(args)


    main(args)
