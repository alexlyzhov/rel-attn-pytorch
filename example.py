import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import RelAttnTransformer

from torch_geometric.datasets import GNNBenchmarkDataset


train_dataset = GNNBenchmarkDataset('.', 'MNIST', 'train')
loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

model = RelAttnTransformer(
    node_input_dim=train_dataset.num_node_features,
	edge_input_dim=train_dataset.num_edge_features,
	node_output_dim=0,
	edge_output_dim=0,
	graph_output_dim=train_dataset.num_classes,
	node_hidden_size=16,
	edge_hidden_size=1,
	layers=3,
	heads=2,
	node_bottleneck=8,
	edge_bottleneck_1=1,
	edge_bottleneck_2=1,
	dropout_prob=0,
	rel_attn_v2=True,
	attn_dropout_prob=0,
	rel_attn_edge_updates=True,
    model_residual=False,
	graph_output_type='mean_pool',
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device).train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
epochs = 1

for epoch in range(epochs):
	for g in loader:
		g = g.to(device)
		g.edge_attr = g.edge_attr.unsqueeze(1)  # by default we are expecting multidim edge vectors, whereas in this dataset they are 1D
		optimizer.zero_grad()
		node_lvl_output, edge_lvl_output, graph_lvl_output = model(g)
		loss = F.cross_entropy(graph_lvl_output, g.y)
		loss.backward()
		optimizer.step()
