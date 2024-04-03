

# autoencoder with KL loss for voice en- and decoding
# supposed to only encode the voice part of audio input and remove the noise

import torch
import torch.nn as nn

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
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
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
    def __init__(self, n_layers):
        super(VAE, self).__init__()
        self.encoder = Encoder(n_layers)
        self.decoder = Decoder(n_layers)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
