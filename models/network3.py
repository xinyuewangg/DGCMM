import torch
import torch.nn as nn

PRETAINED_PATH = '/data/yilin/dgc.pytorch/ImageNet/ccbn_z2048_lr1e-2_cls10_gmm0.01_rec1.0/model_best_ep047_acc75.61_gma0.86.pth'


class Network3(nn.Module):
    def __init__(self, encoder, decoder, num_classes, h_dim, z_dim, generative=False, mlp_dim=None, pretrained=False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generative = generative
        assert h_dim == 2048
        if generative:
            h_dim = h_dim + num_classes
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_logvar = nn.Linear(h_dim, z_dim)

        if pretrained:
            state_dict = torch.load(PRETAINED_PATH, map_location='cpu')
            state_dict = {k[7:]: v for k, v in state_dict['state_dict'].items()}  # remove "module." prefix in keys
            state_dict = {k: v for k, v in state_dict.items() if
                          k[:11] not in ('decoder.bn6', 'decoder.bn7', 'decoder.bn8')}
            del state_dict['classifier.weight']
            del state_dict['classifier.bias']
            self.load_state_dict(state_dict, strict=False)

        if mlp_dim is None:
            self.classifier = nn.Linear(z_dim, num_classes)
        else:
            self.classifier = nn.Sequential(nn.Linear(z_dim, mlp_dim),
                                            nn.ReLU(True),
                                            nn.Linear(mlp_dim, num_classes))

    def forward(self, x, y):
        mu, logvar, z = self.get_z(x, y)
        logits = self.classifier(z)
        x_reconst = self.decode(z, y)

        return mu, logvar, z, logits, x_reconst

    def reparameterize(self, mu, logvar):
        if self.training:
            std = (logvar.clamp(-50, 50).exp() + 1e-8) ** 0.5
            eps = torch.randn_like(logvar)
            return eps * std + mu
        else:
            return mu

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x, y):
        return self.decoder(x, y)

    def get_z(self, x, y=None):
        if self.generative:
            assert y is not None
        h = self.encode(x)
        if self.generative:
            h = torch.cat([h, y], dim=-1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z

    def get_logits(self, x, y=None):
        mu, logvar, z = self.get_z(x, y)
        logits = self.classifier(z)
        return logits
