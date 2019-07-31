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

features = torch.randn((1, 5)) # creates a random vector/tensor 1x5
weights = torch.randn_like(features) # creates a random vector 1x5 (same size as features)
bias = torch.randn((1, 1))

# y = f(w*x + b)
# y = activation(weights*features + bias)

y = activation(torch.mm(features, weights.view(5, 1)) + bias)

print(y)