import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from PIL import Image
import imageio

from vae import VAE
from latents import Latents
from loss import SDSLoss

def save_image(iteration, output, cfg):
  np_output = output.to('cpu').detach().numpy().copy()
  img_output = np.reshape(np_output[0], (28, 28))

  img_output_path = os.path.join(cfg.output_dir, f'output-{iteration}.jpg')
  plt.imshow(img_output)
  plt.savefig(img_output_path)
  plt.close()

def save_gif(outputs, cfg):
  outputs_np = []
  for output in outputs:
    np_output = output.to('cpu').detach().numpy().copy()
    img_output = np.reshape(np_output[0], (28, 28))
    outputs_np.append(img_output)

  # Save gif
  output_path = os.path.join(cfg.output_dir, 'output.gif')
  imageio.mimsave(
      output_path,
      outputs_np,
      format="GIF",
      duration=0.1,
      loop=0
  )

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Prompt Image Editor')
  parser.add_argument('--input_image_path', type=str, default='assets/image-0.png', help='Path to the input image')
  parser.add_argument('--output_dir', type=str, default='output/', help='Path to the output directory')
  parser.add_argument('--prompt', type=str, default='high heel seen from the side', help='Prompt for image generation')
  parser.add_argument('--iterations', type=int, default=10000, help='Number of iterations for optimization')
  parser.add_argument('--save_iterations', type=int, default=100, help='Iterations to save images')

  cfg = parser.parse_args()
  input_image_path = cfg.input_image_path
  output_dir = cfg.output_dir
  prompt = cfg.prompt
  iterations = cfg.iterations
  save_iterations = cfg.save_iterations

  os.makedirs(output_dir, exist_ok=True)

  # Load device and model
  device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
  model = VAE().to(device)
  model.load_state_dict(torch.load('model_weights.pth'))

  # load input image
  input = Image.open(input_image_path).convert('L')
  input = transforms.ToTensor()(input)
  input = transforms.Resize((28, 28))(input)
  input = input.unsqueeze(1).to(device)
  output, z, mu, logvar = model(input)

  latents = Latents(z).to(device)
  sds_loss = SDSLoss(prompt, device).to(device)
  optimizer = torch.optim.Adam(latents.parameters(), lr=0.01)

  # update latents with the prompt
  latents.train()
  progress_bar = tqdm(range(0, iterations))
  outputs = []
  for iteration in range(iterations):
    cur_latents = latents()
    output = model.decoder(cur_latents)

    loss = sds_loss(output)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if iteration % save_iterations == 0:
      save_image(iteration, output, cfg)
      outputs.append(output)

    if iteration % 10 == 0:
      progress_bar.set_postfix({"Loss": loss.cpu().detach().numpy()})
      progress_bar.update(10)
    if iteration == iterations:
      progress_bar.close()

  # Save the final output
  save_gif(outputs, cfg)
