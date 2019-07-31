import torch

def activation(x):
	"""
		Sigmoid activation function

		Arguments
		---------
		x: torch.Tensor
	"""
	return 1/(1 + torch.exp(-x))

torch.manual_seed(7) # rgn seed for randn

features = torch.randn((1, 3)) # creates a random vector/tensor 1x5
n_input = features.shape[1]
n_hidden = 2
n_output = 1

# weightd
W1 = torch.randn(n_input, n_hidden)
W2 = torch.randn(n_hidden, n_output)

# bias
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

# y = f(w*x + b)
# y = activation(weights*features + bias)

h = activation(torch.mm(features, W1) + B1)
y = activation(torch.mm(h, W2) + B2)

print(y)
