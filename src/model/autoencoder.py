

# autoencoder with KL loss for voice en- and decoding
# supposed to only encode the voice part of audio input and remove the noise

import torch
import torch.nn as nn
import torch.nn.functional as F

class DownLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DownLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x
    
class UpLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(UpLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x
    
class MidLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(MidLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x
    
# VAE Encoder
class Encoder(nn.Module):
    def __init__(self, n_layers):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.down_layers = nn.ModuleList()
        self.down_layers.append(DownLayer(1, 16, 3, 1, 1))
        for i in range(n_layers-1):
            self.down_layers.append(DownLayer(16*(2**i), 16*(2**(i+1)), 3, 1, 1))
        self.mid_layer = MidLayer(16*(2**(n_layers-1)), 16*(2**(n_layers-1)), 3, 1, 1)

    def forward(self, x):
        for i in range(self.n_layers):
            x = self.down_layers[i](x)
        x = self.mid_layer(x)
        return x
    
# VAE Decoder
class Decoder(nn.Module):
    def __init__(self, n_layers):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.up_layers = nn.ModuleList()
        for i in range(n_layers-1):
            self.up_layers.append(UpLayer(16*(2**(n_layers-i)), 16*(2**(n_layers-i-1)), 3, 1, 1))
        self.up_layers.append(UpLayer(16*2, 16, 3, 1, 1))
        self.out_layer = nn.Conv1d(16, 1, 3, 1, 1)
    
    def forward(self, x):
        for i in range(self.n_layers):
            x = self.up_layers[i](x)
        x = self.out_layer(x)
        return x
    
# VAE
class VAE(nn.Module):
    def __init__(self, n_layers, deterministic=False):
        super(VAE, self).__init__()
        self.encoder = Encoder(n_layers)
        self.decoder = Decoder(n_layers)
        self.deterministic = deterministic
    
    def forward(self, x):
        # encode
        x = self.encoder(x)

        
        mean, logvar = x.chunk(2, dim=1)
        if not self.deterministic:
            # sample from the latent space
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            x = mean + eps*std
        else:
            # use mean as the latent space
            x = mean

        # decode
        x = self.decoder(x)
        return x
    
    def loss(self, x, target):
        # encode
        x = self.encoder(x)
        mean, logvar = x.chunk(2, dim=1)
        
        # KL loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        
        if not self.deterministic:
            # sample from the latent space
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            x = mean + eps*std
        else:
            # use mean as the latent space
            x = mean

        # decode
        x = self.decoder(x)
        
        # reconstruction loss
        recon_loss = nn.MSELoss()(x, target)
        
        return kl_loss + recon_loss
    
def test():
    vae = VAE(3)
    x = torch.randn(1, 1, 1000)
    target = torch.randn(1, 1, 1000)
    loss = vae.loss(x, target)
    print(loss)
