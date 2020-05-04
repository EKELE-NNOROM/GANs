import torch.nn as nn
import torch.nn.functional as F
import torch

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


#########################################################
#                       U-NET
#########################################################

class Encoder(nn.Module):
    def __init__(self, n_input, n_output, normalize=True, dropout=0.0):
        super(Encoder, self).__init__()
        layers = [nn.Conv2d(n_input, n_output, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(n_output))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, n_input, n_output, dropout=0.0):
        super(Decoder, self).__init__()
        layers = [
            nn.ConvTranspose2d(n_input, n_output, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(n_output),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x



class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = Encoder(in_channels, 64, normalize=False)
        self.down2 = Encoder(64, 128)
        self.down3 = Encoder(128, 256)
        self.down4 = Encoder(256, 512, dropout=0.5)
        self.down5 = Encoder(512, 512, dropout=0.5)
        self.down6 = Encoder(512, 512, dropout=0.5)
        self.down7 = Encoder(512, 512, dropout=0.5)
        self.down8 = Encoder(512, 512, normalize=False, dropout=0.5)

        self.up1 = Decoder(512, 512, dropout=0.5)
        self.up2 = Decoder(1024, 512, dropout=0.5)
        self.up3 = Decoder(1024, 512, dropout=0.5)
        self.up4 = Decoder(1024, 512, dropout=0.5)
        self.up5 = Decoder(1024, 256)
        self.up6 = Decoder(512, 128)
        self.up7 = Decoder(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


##############################################################
#                   Discriminator
#############################################################

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
                                   *discriminator_block(in_channels * 2, 64, normalize=False),
                                   *discriminator_block(64, 128),
                                   *discriminator_block(128, 256),
                                   *discriminator_block(256, 512),
                                   nn.ZeroPad2d((1, 0, 1, 0)),
                                   nn.Conv2d(512, 1, 4, padding=1, bias=False)

        )

    def forward(self, img_A, img_B):
        # concatenate image and condition image by channels to produce generated input
        gen_input = torch.cat((img_A, img_B), 1)
        return self.model(gen_input)
