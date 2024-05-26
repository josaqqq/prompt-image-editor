from sched import scheduler
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from vae import VAE
from latents import Latents
from loss import SDSLoss

def save_image(iteration, output):
  np_output = output.to('cpu').detach().numpy().copy()
  img_output = np.reshape(np_output[0], (28, 28))

  plt.imshow(img_output)
  plt.savefig('./output/iteration-{}.jpg'.format(iteration))
  plt.close()

if __name__ == '__main__':
  # load dataset
  dataset = FashionMNIST('./data',
                         train=True,
                         download=True,
                         transform=transforms.ToTensor())
  
  prompt = 'high heel seen from the side'
  iterations = 10000
  save_iterations = 100
  
  device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
  model = VAE().to(device)
  model.load_state_dict(torch.load('model_weights.pth'))

  # prepare initial latents
  input, _ = dataset[0]
  input = input.unsqueeze(1).to(device)
  output, z, mu, logvar = model(input)
  
  latents = Latents(z).to(device)
  sds_loss = SDSLoss(prompt, device).to(device)
  optimizer = torch.optim.RAdam(latents.parameters(), lr=0.01)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

  # update latents with the prompt
  latents.train()
  progress_bar = tqdm(range(0, iterations))
  for iteration in range(iterations):
    cur_latents = latents()
    output = model.decoder(cur_latents)

    loss = sds_loss(output)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if iteration % save_iterations == 0:
      save_image(iteration, output)

    if iteration % 10 == 0:
      progress_bar.set_postfix({"Loss": loss.cpu().detach().numpy()})
      progress_bar.update(10)
    if iteration == iterations:
      progress_bar.close()
