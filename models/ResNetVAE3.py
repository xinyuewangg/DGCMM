# CCBN + 2048
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.hub import load_state_dict_from_url

PRETAINED_URL = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
from models.layers import ccbn


class ResNet_VAE_encoder(nn.Module):
    def __init__(self, fc_hidden1=1024, h_dim=768, pretrained=False):
        super(ResNet_VAE_encoder, self).__init__()

        # self.fc_hidden1, self.fc_hidden2 = fc_hidden1, h_dim
        # encoding components
        resnet = models.resnet50(pretrained=False)
        if pretrained:
            state_dict = load_state_dict_from_url(PRETAINED_URL)
            # state_dict = {k[7:]: v for k, v in state_dict['state_dict'].items()}  # remove "module." prefix in keys
            del state_dict['fc.weight']
            del state_dict['fc.bias']
            resnet.load_state_dict(state_dict, strict=False)

        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)

        # self.fc1 = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
        # self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        # self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        # self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y=None):
        x = self.resnet(x)  # ResNet
        x = x.view(x.size(0), -1)  # flatten output of conv
        return x


class ResNet_VAE_decoder(nn.Module):
    def __init__(self, num_classes, h_dim=768, z_dim=256, img_size=224, norm_layer=None):
        super(ResNet_VAE_decoder, self).__init__()
        if norm_layer is None:
            bn = functools.partial(ccbn,
                                   which_linear=(functools.partial(nn.Linear, bias=False)),
                                   cross_replica=False,
                                   mybn=False,
                                   input_size=num_classes)
            self.norm_layer = bn
        self.img_size = img_size
        self.h_dim, self.z_dim = h_dim, z_dim
        self.ch = 128
        self.ks = 3  # 2d kernal size
        self.ss = 2  # 2d strides
        self.pd = 1  # 2d padding

        self.relu = nn.ReLU(inplace=True)
        # Sampling vector
        hidden_dim = self.ch * 6 * 6

        self.fc4 = nn.Linear(self.z_dim, hidden_dim)
        self.fc_bn4 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)

        # Decoder
        self.convTrans6 = nn.ConvTranspose2d(in_channels=self.ch, out_channels=self.ch // 2,
                                             kernel_size=self.ks, stride=self.ss,
                                             padding=self.pd)
        self.bn6 = self.norm_layer(self.ch // 2, momentum=0.01)
        self.convTrans7 = nn.ConvTranspose2d(in_channels=self.ch // 2, out_channels=self.ch // 4,
                                             kernel_size=self.ks, stride=self.ss,
                                             padding=self.pd)
        self.bn7 = self.norm_layer(self.ch // 4, momentum=0.01)

        self.convTrans8 = nn.ConvTranspose2d(in_channels=self.ch // 4, out_channels=3,
                                             kernel_size=self.ks, stride=self.ss,
                                             padding=self.pd)
        self.bn8 = self.norm_layer(3, momentum=0.01)
        self.sigmoid = nn.Sigmoid()  # y = (y1, y2, y3) \in [0 ,1]^3

    def forward(self, z, y):
        x = self.relu(self.fc_bn4(self.fc4(z))).view(-1, self.ch, 6, 6)
        x = self.convTrans6(x)
        x = self.bn6(x, y)
        x = self.relu(x)
        x = self.convTrans7(x)
        x = self.bn7(x, y)
        x = self.relu(x)
        x = self.convTrans8(x)
        x = self.bn8(x, y)
        x = self.sigmoid(x)
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        return x
