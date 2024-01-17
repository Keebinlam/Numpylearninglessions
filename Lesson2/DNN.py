import torch
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import jovian

dataset = MNIST(root='data/', download=True, transform=ToTensor())
# by default, the images are downloaded as PIL, so using "transform=ToTensor) we are making sure the file can be read by tensor

# the image can be viewed, and we can get a good understand of data being managed. Tensor will set up the image to 1,28,28. 1 being color value, if it was RGB, it would be 3. 28 by 28 pixal in size. We can use plt.imshow to isplace
# we need to adjust the order of the dimension first. Since we need the color value at the end. We will use permute to do this
# we are getting the image and label for the first data value
image, label = dataset[0]
print('image.shape:', image.shape)  # printing to confirm the dimensions
# showing the image, but also remaping the order
plt.imshow(image.permute(1, 2, 0), cmap='gray')
print('Label:', label)
