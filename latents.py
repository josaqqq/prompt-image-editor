import torch
from torch import nn

class Latents(nn.Module):
  def __init__(self, init_values):
    super(Latents, self).__init__()
    self.latents = nn.Parameter(init_values)

  def forward(self):
    return self.latents
