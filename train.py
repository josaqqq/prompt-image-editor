import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import numpy as np
import matplotlib.pyplot as plt

from vae import VAE
from loss import VAELoss

def load_dataset():
  BATCH_SIZE = 100
  dataset = FashionMNIST('./data',
                        train=True,
                        download=True,
                        transform=transforms.ToTensor())
  
  train_size = int(len(dataset) * 0.8)
  test_size = int(len(dataset) * 0.2)
  train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

  train_loader = DataLoader(dataset=train_data,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=0)
  test_loader = DataLoader(dataset=test_data,
                           batch_size=BATCH_SIZE,
                           shuffle=True,
                           num_workers=0)

  return train_loader, test_loader

def save_image(epoch, input, output, mode):
  np_input = input.to('cpu').detach().numpy().copy()
  img_input = np.reshape(np_input[0], (28, 28))
  np_output = output.to('cpu').detach().numpy().copy()
  img_output = np.reshape(np_output[0], (28, 28))

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
  ax1.imshow(img_input)
  ax2.imshow(img_output)
  plt.savefig('./output/epoch-{}-{}.jpg'.format(mode, epoch + 1))
  plt.close()

def train(train_loader, test_loader):
  num_epochs = 20

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  model = VAE().to(device)
  vae_loss = VAELoss().to(device)
  optimizer = torch.optim.RAdam(model.parameters(), lr=0.01)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

  for epoch in range(num_epochs):
    for i, (x, _) in enumerate(train_loader):
      input = x.to(device).to(torch.float32)
      output, z, mu, logvar = model(input)
      loss = vae_loss(output, input, mu, logvar)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      scheduler.step()

      if i == 0:
        save_image(epoch, input, output, 'train')

      if (i + 1) % 50 == 0:
        print('Epoch: {}, Loss: {}'.format(epoch + 1, loss))

    with torch.no_grad():
      for i, (x, _) in enumerate(test_loader):
        input = x.to(device).to(torch.float32)
        output, z, mu, logvar = model(input)
        loss = vae_loss(output, input, mu, logvar)

        if i == 0:
          save_image(epoch, input, output, 'test')
      
      print('Epoch: {}, Test Loss: {}'.format(epoch + 1, loss))
    
  # save the model weights
  torch.save(model.state_dict(), 'model_weights.pth')

if __name__ == '__main__':
  train_loader, test_loader = load_dataset()
  train(train_loader, test_loader)
