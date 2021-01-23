# gcn_model.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GENConv
from gcn import add_features_, dataloader
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import ClusterData, ClusterLoader
from sklearn.metrics import mean_squared_error, r2_score


num_epochs = 20
data_, G = dataloader()
data_ = add_features_(data_, G)
dataset = data_
print(dataset)
# dataset = InMemoryDataset.collate(data)
cluster_data = ClusterData(data_, num_parts=50, recursive=False)
test_mask = cluster_data
train_loader = ClusterLoader(cluster_data, batch_size=5, shuffle=True,
							 num_workers=12)

class Net(torch.nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = GCNConv(dataset.num_node_features, 32)
		# self.conv2 = GCNConv(16, dataset.num_classes)
		self.conv3 = GCNConv(32, 16)
		self.conv2 = GCNConv(16, dataset.y.shape[1])

	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		x = self.conv1(x, edge_index)
		x = F.relu(x)
		x = self.conv3(x, edge_index)
		x = F.relu(x)
		x = F.dropout(x, training=self.training)
		x = self.conv2(x, edge_index)

		return F.relu(x)
		# return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = data_.to(device)


test_mask = data_
optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-1)

def train():
	model.train()
	loss_all=0
	for batch in train_loader:
		batch = batch.to(device)
		optimizer.zero_grad()
		out = model(data)
		loss = F.mse_loss(out, data.y)
		loss.backward()
		loss_all +=  loss.item()
		optimizer.step()
	print(loss_all)


for epoch in range(num_epochs):
	train()



model.eval()
pred = model(data)
# print(data.y.shape)
# print(pred.shape)
mse = mean_squared_error(data.y.detach().numpy(), pred.detach().numpy(), multioutput='uniform_average')
r_square = r2_score(data.y.detach().numpy(), pred.detach().numpy(), multioutput= 'uniform_average')
print(mse)
print(r_square)
