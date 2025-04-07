import os
from tqdm import tqdm
import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import numpy as np
import matplotlib.pyplot as plt

from vae import VAE
from loss import VAELoss

def load_dataset(cfg):
  dataset = FashionMNIST('./data',
                        train=True,
                        download=True,
                        transform=transforms.ToTensor())

  train_size = int(len(dataset) * 0.8)
  test_size = int(len(dataset) * 0.2)
  train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

  train_loader = DataLoader(dataset=train_data,
                            batch_size=cfg.batch_size,
                            shuffle=True,
                            num_workers=0)
  test_loader = DataLoader(dataset=test_data,
                           batch_size=cfg.batch_size,
                           shuffle=True,
                           num_workers=0)

  return train_loader, test_loader

def save_image(epoch, input, output, mode, cfg):
  np_input = input.to('cpu').detach().numpy().copy()
  img_input = np.reshape(np_input[0], (28, 28))
  np_output = output.to('cpu').detach().numpy().copy()
  img_output = np.reshape(np_output[0], (28, 28))

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
  ax1.imshow(img_input)
  ax2.imshow(img_output)

  output_dir = os.path.join(cfg.output_dir, mode)
  os.makedirs(output_dir, exist_ok=True)
  output_path = os.path.join(cfg.output_dir, mode, f'epoch-{epoch + 1}.jpg')
  plt.savefig(output_path)
  plt.close()

def train(train_loader, test_loader, cfg):
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  model = VAE().to(device)
  vae_loss = VAELoss().to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

  losses_training = []
  losses_test = []
  for epoch in tqdm(range(cfg.num_epochs)):
    for i, (x, _) in enumerate(train_loader):
      input = x.to(device).to(torch.float32)
      output, z, mu, logvar = model(input)
      loss = vae_loss(output, input, mu, logvar)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      losses_training.append(loss.item())
      if i == 0:
        save_image(epoch, input, output, 'train', cfg)

    with torch.no_grad():
      for i, (x, _) in enumerate(test_loader):
        input = x.to(device).to(torch.float32)
        output, z, mu, logvar = model(input)
        loss = vae_loss(output, input, mu, logvar)

        losses_test.append(loss.item())
        if i == 0:
          save_image(epoch, input, output, 'test', cfg)

      print('Epoch: {}, Test Loss: {}'.format(epoch + 1, loss))

  # save the training and test losses
  np.save(os.path.join(cfg.output_dir, 'losses_training.npy'), np.array(losses_training))
  np.save(os.path.join(cfg.output_dir, 'losses_test.npy'), np.array(losses_test))

  # plot the training and test losses
  plt.plot(losses_training, label='Training Loss')
  plt.xlabel('Iterations')
  plt.ylabel('Loss')
  plt.title('Training Losses')
  plt.legend()
  plt.savefig(os.path.join(cfg.output_dir, 'training_losses.png'))
  plt.close()
  plt.clf()

  plt.plot(losses_test, label='Test Loss')
  plt.xlabel('Iterations')
  plt.ylabel('Loss')
  plt.title('Test Losses')
  plt.legend()
  plt.savefig(os.path.join(cfg.output_dir, 'test_losses.png'))
  plt.close()
  plt.clf()

  # save the model weights
  torch.save(model.state_dict(), cfg.model_weights_path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Prompt Image Editor')
  parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
  parser.add_argument('--output_dir', type=str, default='output/training', help='Path to the output directory')
  parser.add_argument('--model_weights_path', type=str, default='model_weights.pth', help='Path to save model weights')
  parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
  parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for optimizer')

  cfg = parser.parse_args()

  output_dir = cfg.output_dir
  os.makedirs(output_dir, exist_ok=True)

  train_loader, test_loader = load_dataset(cfg)
  train(train_loader, test_loader, cfg)
