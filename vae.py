import torch
from torch import nn
import torch.nn.functional

z_dim = 30

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=2, padding='same')
    self.conv2 = nn.Conv2d(32, 32, kernel_size=2, padding='same')
    self.conv3 = nn.Conv2d(32, 32, kernel_size=2, padding='same')
    self.lr1 = nn.Linear(32*3*3, 100)
    self.lr_mu = nn.Linear(100, z_dim)
    self.lr_logvar = nn.Linear(100, z_dim)
    self.bn = nn.BatchNorm2d(32)
    self.relu = nn.ReLU()
    self.maxpool_2D = nn.AvgPool2d(2, stride=2)
  
  def forward(self, x):
    x = self.conv1(x)
    x = self.bn(x)
    x = self.relu(x)
    x = self.maxpool_2D(x)

    x = self.conv2(x)
    x = self.bn(x)
    x = self.relu(x)
    x = self.maxpool_2D(x)

    x = self.conv3(x)
    x = self.bn(x)
    x = self.relu(x)
    x = self.maxpool_2D(x)

    x = x.view(-1, 32*3*3)
    x = self.lr1(x)
    x = self.relu(x)

    mu = self.lr_mu(x)
    logvar = self.lr_logvar(x)
    ep = torch.randn_like(mu)
    z = mu + torch.exp(logvar / 2)*ep

    return z, mu, logvar

class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    self.lr1 = nn.Linear(z_dim, 8*3*3)
    self.relu = nn.ReLU()
    self.uc1 = nn.ConvTranspose2d(8, 32, kernel_size=2, stride=2, output_padding=1)
    self.uc2 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
    self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding='same')
    self.bn = nn.BatchNorm2d(32)
    self.conv2 = nn.Conv2d(32, 1, kernel_size=1)

  def forward(self, x):
    x = self.lr1(x)
    x = self.relu(x)
    x = x.view(-1, 8, 3, 3)

    x = self.uc1(x)
    x = self.conv1(x)
    x = self.bn(x)
    x = self.relu(x)
    
    x = self.uc2(x)
    x = self.conv1(x)
    x = self.bn(x)
    x = self.relu(x)
    
    x = self.uc2(x)
    x = self.conv1(x)
    x = self.bn(x)
    x = self.relu(x)
    
    x = self.conv2(x)
    x = torch.sigmoid(x)    
    
    return x
    
class VAE(nn.Module):
  def __init__(self):
    super(VAE, self).__init__()
    self.encoder = Encoder()
    self.decoder = Decoder()

  def forward(self, x):
    z, mu, logvar = self.encoder(x)
    x = self.decoder(z)
    return x, z, mu, logvar
