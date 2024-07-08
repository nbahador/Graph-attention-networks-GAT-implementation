# -------------------------------------------------------- #
# -------------------------------------------------------- #
# implementing a graph neural network with an
# attention mechanism from scratch for attributes classification
# -------------------------------------------------------- #
# -------------------------------------------------------- #

import numpy as np
import torch

from scipy.io import loadmat
annots = loadmat('adj_average_final.mat')
mat=annots['adj_average_final']
adj_matrix_0 = annots['adj_average_final']
adj_matrix = annots['adj_average_final']


annots = loadmat('shuffled_input_features.mat')
mat=annots['shuffled_input_features']
shuffled_input_features_0 = annots['shuffled_input_features']
x = torch.tensor(annots['shuffled_input_features'])


annots = loadmat('shuffled_targets.mat')
mat=annots['shuffled_targets']
shuffled_targets_0 = annots['shuffled_targets']
shuffled_targets_0 = np.where(shuffled_targets_0 == 0, 0.01, shuffled_targets_0)
shuffled_targets_0 = np.where(shuffled_targets_0 == 1, 0.95, shuffled_targets_0)
y = torch.tensor(shuffled_targets_0.T)

edge_index = torch.tensor(adj_matrix).nonzero().t().contiguous()

# Convert the adjacency matrix to a sparse format
adj_matrix = torch.tensor(adj_matrix)
# Extract the indices of non-zero elements
indices = torch.where(adj_matrix != 0)
# Extract the non-zero edge weights
edge_weights = adj_matrix[indices[0], indices[1]]




# ---------------------------------------------------------- #
# ---------------------------------------------------------- #
# ---------------------------------------------------------- #

import torch
from torch_geometric.datasets import Planetoid

# Import dataset from PyTorch Geometric
#dataset = Planetoid(root=".", name="CiteSeer")

#xxx = dataset.data.val_mask
#xx = dataset.data.edge_index


# Print information about the dataset
#print(f'Number of nodes: {x.shape[0]}')



from torch_geometric.utils import degree
from collections import Counter



# Get list of degrees for each node
n = len(adj_matrix)
degrees = []
for i in range(n):
    degree = sum(adj_matrix[i]) + sum(adj_matrix[j][i] for j in range(n))
    degrees.append(degree)

# Count the number of nodes for each degree
numbers = Counter(degrees)

import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, GATv2Conv


class GATLayer(torch.nn.Module):
  def __init__(self, in_features, out_features, dropout, alpha, concat=True):
    super(GATLayer, self).__init__()
    self.dropout = dropout  # drop prob = 0.6
    self.in_features = in_features  #
    self.out_features = out_features  #
    self.alpha = alpha  # LeakyReLU with negative input slope, alpha = 0.2
    self.concat = concat  # conacat = True for all layers except the output layer.
    # Xavier Initialization of Weights
    # Alternatively use weights_init to apply weights of choice
    self.W = torch.nn.Parameter(torch.zeros(size=(in_features, out_features)))
    torch.nn.init.xavier_uniform_(self.W.data, gain=1.414)
    self.a = torch.nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
    torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)
    # LeakyReLU
    self.leakyrelu = torch.nn.LeakyReLU(self.alpha)
    self.W2 = torch.nn.Parameter(torch.zeros(size=(in_features, 1)))
    torch.nn.init.xavier_uniform_(self.W2.data, gain=1.414)
    self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

  def forward(self, input, adj):
    # Linear Transformation
    h = input * self.W  # matrix multiplication
    N = h.size()[0]
    # Attention Mechanism
    a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
    mul_a = torch.matmul(a_input.double(), self.a.double())
    e = self.leakyrelu(mul_a.squeeze(2))
    # Masked Attention
    zero_vec = -9e15 * torch.ones_like(e)
    attention = torch.where(adj > 0, e, zero_vec)
    attention = F.softmax(attention, dim=1)
    self.attention = F.dropout(attention, self.dropout, training=self.training)
    h_prime = torch.matmul(self.attention, h)
    h_prime_2 = torch.matmul(adj, self.W2.double())
    h_prime_3 = torch.matmul(h_prime_2.T,input)
    h_prime_3 = h_prime_3.T
    if self.concat:
      return F.elu(h_prime_3)
    else:
      return h_prime_3


fea_in = x
edge_weights = edge_weights.double()
criterion = torch.nn.MSELoss()
model = GATLayer(64, 38, 0.001, 0.01)
optimizer = model.optimizer

y_pred = torch.zeros(len(x[0, :]),1)

criterion = torch.nn.MSELoss()

model.train()
epochs = 0
# Train the model for multiple epochs
while True:
  epochs += 1
  model.train()
  h = model(fea_in, adj_matrix)
  #log_softmax = torch.log_softmax(h, dim=0)
  #loss = -torch.sum(log_softmax * y, dim=1)
  #loss = loss.mean()
  loss = criterion(h, y)

  # Backward pass and optimization
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  threshold = 1e-5
  grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), threshold)

  y_pred = h

  if (epochs % 10 == 0):
    print(f' Epoch {epochs} | Loss Max: {np.max(loss.item()):.3f}')

  if grad_norm < threshold:
    print("Gradient norm below threshold. Training stopped.")
    break

  if loss.item() < 0.02:
    print("Loss below threshold. Training stopped.")
    break


import scipy.io
# Save the variables to a .mat file
y_pred = y_pred.detach().numpy()
scipy.io.savemat('y_pred.mat', {'y_pred': y_pred})
W_weights_final = model.W.detach().numpy()
scipy.io.savemat('W_weights_final.mat', {'W_weights_final': W_weights_final})
W2_weights_final = model.W2.detach().numpy()
scipy.io.savemat('W2_weights_final.mat', {'W2_weights_final': W2_weights_final})
attention_weights_final = model.attention.detach().numpy()
scipy.io.savemat('attention_weights_final.mat', {'attention_weights_final': attention_weights_final})




print("Gradient norm below threshold. Training stopped.")











