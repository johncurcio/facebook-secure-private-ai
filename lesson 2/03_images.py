#matplotlib inline
#config InlineBackend.figure_format = 'retina'

import torch 

from torchvision import datasets, transforms

def activation(x):
	"""
		Sigmoid activation function

		Arguments
		---------
		x: torch.Tensor
	"""
	return 1/(1 + torch.exp(-x))

transform = transforms.Compose([transforms.ToTensor(), transforms.normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Get our data
images, labels = next(iter(trainloader))
# Flatten images
inputs = images.view(images.shape[0], -1)

w1 = torch.randn(784, 256)
b1 = torch.randn(256)

w2 = torch.randn(256, 10)
b2 = torch.randn(10)

h = activation(torch.mm(inputs, w1) + b1)

out = torch.mm(h, w2) + b2