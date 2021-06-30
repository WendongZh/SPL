import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import *
import torch.nn.functional as F

from rn import RN_B, RN_L, SPADE


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, in_channels, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning

class G_Net(nn.Module):
    def __init__(self, input_channels, residual_blocks, threshold):
        super(G_Net, self).__init__()

        self.G0 = 256
        self.G = 32
        self.D = 16
        self.C = 6

        # Encoder
        self.encoder_prePad = nn.ReflectionPad2d(3)
        self.encoder_conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=7, padding=0)
        self.encoder_in1 = RN_B(feature_channels=64)
        self.encoder_relu1 = nn.ReLU(True)
        self.encoder_conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.encoder_in2 = RN_B(feature_channels=128)
        self.encoder_relu2 = nn.ReLU(True)
        self.encoder_conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.encoder_in3 = RN_B(feature_channels=256)
        self.encoder_relu3 = nn.ReLU(True)

        # Middle
        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock_Spade(256, layout_dim=256, dilation=2, use_spectral_norm=False)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        # Encoder semantic branch
        self.encoder_prePad_sm = nn.ReflectionPad2d(3)
        self.encoder_conv1_sm = nn.Conv2d(in_channels=input_channels+1, out_channels=64, kernel_size=7, padding=0)
        self.encoder_relu1_sm = nn.LeakyReLU(0.1)
        self.encoder_conv2_sm = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.encoder_relu2_sm = nn.LeakyReLU(0.1)
        self.encoder_conv3_sm = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.encoder_relu3_sm = nn.LeakyReLU(0.1)
        self.encoder_conv4_sm = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.encoder_relu4_sm = nn.LeakyReLU(0.1)

        self.encoder_sm_out = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        # branch for Asl feature recon
        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G0, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G0 * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2),
        )
        self.feature_recon = nn.Sequential(
            ResnetBlock(256, dilation=1, use_spectral_norm=False),
            ResnetBlock(256, dilation=1, use_spectral_norm=False),
            ResnetBlock(256, dilation=1, use_spectral_norm=False),
            ResnetBlock(256, dilation=1, use_spectral_norm=False),
        )
        self.feature_mapping = nn.Conv2d(in_channels=256, out_channels=152, kernel_size=1, stride=1, padding=0)


        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128*4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            RN_L(128),
            nn.ReLU(True),

            nn.Conv2d(128, 64*4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            RN_L(64),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=input_channels, kernel_size=7, padding=0)

        )


    def encoder(self, x, mask):
        x = self.encoder_prePad(x)

        x = self.encoder_conv1(x)
        x = self.encoder_in1(x, mask)
        x = self.encoder_relu2(x)

        x = self.encoder_conv2(x)
        x = self.encoder_in2(x, mask)
        x = self.encoder_relu2(x)

        x = self.encoder_conv3(x)
        x = self.encoder_in3(x, mask)
        x = self.encoder_relu3(x)
        return x

    def encoder_sm(self, x):
        x = self.encoder_prePad_sm(x)

        x = self.encoder_conv1_sm(x)
        x = self.encoder_relu2_sm(x)

        x = self.encoder_conv2_sm(x)
        x = self.encoder_relu2_sm(x)

        x = self.encoder_conv3_sm(x)
        x = self.encoder_relu3_sm(x)

        x = self.encoder_conv4_sm(x)
        x = self.encoder_relu4_sm(x)
        return x

    def forward(self, x, mask, masked_512, mask_512):
        gt = x
        x_input = (x * (1 - mask).float()) + mask
        # input mask: 1 for hole, 0 for valid
        x = self.encoder(x_input, mask)

        # perform feature recon
        x_sm = self.encoder_sm(torch.cat((masked_512, mask_512), 1))
        x_sm_skip = self.encoder_sm_out(x_sm)
        local_features = []
        for i in range(self.D):
            x_sm_skip = self.rdbs[i](x_sm_skip)
            local_features.append(x_sm_skip)

        x_sm = self.gff(torch.cat(local_features, 1)) + x_sm
        layout = self.feature_recon(x_sm)
        feature_recon = self.feature_mapping(layout)

        # decode the image with layout
        for i in range(len(self.middle)):
            sub_block = self.middle[i]
            x = sub_block(x, layout)

        x = self.decoder(x)

        x = (torch.tanh(x) + 1) / 2

        return x, feature_recon


# original D
class D_Net(nn.Module):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True):
        super(D_Net, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )


    def forward(self, x):

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]




class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=True):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=256, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


class ResnetBlock_Spade(nn.Module):
    def __init__(self, dim, layout_dim, dilation, use_spectral_norm=True):
        super(ResnetBlock_Spade, self).__init__()
        self.conv_block = nn.Sequential(
            SPADE(dim, layout_dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=256, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),

            SPADE(256, layout_dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(dim, track_running_stats=False),
            # RN_L(feature_channels=dim, threshold=threshold),
        )

    def forward(self, x, layout):
        # out = x + self.conv_block(x)
        out = x
        for i in range(len(self.conv_block)):
            sub_block = self.conv_block[i]
            if i == 0 or i == 4:
                out = sub_block(out, layout)
            else:
                out = sub_block(out)

        out_final = out + x
        # skimage.io.imsave('block.png', out[0].detach().permute(1,2,0).cpu().numpy()[:,:,0])

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out_final

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


if __name__ == '__main__':
    print("No Abnormal!")
