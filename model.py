import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn import global_mean_pool

from layers import RelAttnLayer


class RelAttnTransformer(nn.Module):
	def __init__(
			self, node_input_dim, node_hidden_size, edge_input_dim, edge_hidden_size, node_output_dim, edge_output_dim, graph_output_dim, layers, heads, node_bottleneck, edge_bottleneck_1, edge_bottleneck_2, dropout_prob, rel_attn_v2, attn_dropout_prob, rel_attn_edge_updates,model_residual, graph_output_type
		):
		super().__init__()
	
		self.node_hidden_size = node_hidden_size
		self.edge_hidden_size = edge_hidden_size
		self.node_output_dim = node_output_dim
		self.edge_output_dim = edge_output_dim
		self.graph_output_dim = graph_output_dim
		self.layers = layers
		self.heads = heads
		self.node_bottleneck = node_bottleneck
		self.edge_bottleneck_1 = edge_bottleneck_1
		self.edge_bottleneck_2 = edge_bottleneck_2
		self.dropout_prob = dropout_prob
		self.rel_attn_v2 = rel_attn_v2
		self.attn_dropout_prob = attn_dropout_prob
		self.rel_attn_edge_updates = rel_attn_edge_updates
		self.model_residual = model_residual
		self.graph_output_type = graph_output_type

		self.node_input_proj = nn.Linear(node_input_dim, self.node_hidden_size)
		self.edge_input_proj = nn.Linear(edge_input_dim, self.edge_hidden_size)
		self.graph_virtual_node_param = nn.Parameter(torch.randn(1, 1, self.node_hidden_size))
		self.graph_virtual_edge_param = nn.Parameter(torch.randn(1, 1, 1, self.edge_hidden_size))
		self.node_output_proj = nn.Linear(self.node_hidden_size, self.node_output_dim) if self.node_output_dim > 0 else None
		self.edge_output_proj = nn.Linear(self.edge_hidden_size, self.edge_output_dim) if self.edge_output_dim > 0 else None
		self.graph_output_proj = nn.Linear(self.node_hidden_size, self.graph_output_dim) if self.graph_output_dim > 0 else None

		self.layers_module = nn.ModuleList()
		for l in range(self.layers):
			new_layer = RelAttnLayer(
				self.node_hidden_size, self.edge_hidden_size, self.heads,
				self.node_bottleneck, self.edge_bottleneck_1, self.edge_bottleneck_2,
				self.dropout_prob, self.rel_attn_v2, self.attn_dropout_prob)
			self.layers_module.append(new_layer)

	def forward(self, g):
		x = self.node_input_proj(g.x)
		edge_attr = self.edge_input_proj(g.edge_attr)

		node_tensors, dense_node_tensors_mask = to_dense_batch(x, g.batch)
		edge_tensors = to_dense_adj(edge_index=g.edge_index, batch=g.batch, edge_attr=edge_attr)

		if self.graph_output_type == 'core':
			core_node_appendix = self.graph_virtual_node_param.repeat(node_tensors.shape[0], 1, 1)
			node_tensors = torch.cat([node_tensors, core_node_appendix], dim=1)
			core_edge_appendix = self.graph_virtual_edge_param.repeat(edge_tensors.shape[0], edge_tensors.shape[1], 1, 1)
			edge_tensors = torch.cat([edge_tensors, core_edge_appendix], dim=2)
			core_edge_appendix = self.graph_virtual_edge_param.repeat(edge_tensors.shape[0], 1, edge_tensors.shape[2], 1)
			edge_tensors = torch.cat([edge_tensors, core_edge_appendix], dim=1)

		for layer in self.layers_module:
			if self.rel_attn_edge_updates:
				new_node_tensors, new_edge_tensors = layer(node_tensors, edge_tensors)
				if self.model_residual:
					node_tensors = node_tensors + new_node_tensors
					edge_tensors = edge_tensors + new_edge_tensors
				else:
					node_tensors = new_node_tensors
					edge_tensors = new_edge_tensors
			else:
				new_node_tensors, _ = layer(node_tensors, edge_tensors)
				if self.model_residual:
					node_tensors = node_tensors + new_node_tensors
				else:
					node_tensors = new_node_tensors

		if self.graph_output_type == 'core':
			node_tensors = node_tensors[:, :-1, :]
			edge_tensors = edge_tensors[:, :-1, :-1, :]

		x = node_tensors[dense_node_tensors_mask]
		mask = to_dense_adj(g.edge_index, g.batch).unsqueeze(-1).bool()
		edge_attr = torch.masked_select(edge_tensors, mask).view(-1, edge_tensors.shape[-1])

		node_lvl_output = None
		if self.node_output_proj is not None:
			node_lvl_output = self.node_output_proj(x)

		edge_lvl_output = None
		if self.edge_output_proj is not None:
			edge_lvl_output = self.edge_output_proj(edge_attr)

		if self.graph_output_type == 'core':
			graph_lvl_repr = node_tensors[:, -1, :]
		elif self.graph_output_type == 'mean_pool':
			graph_lvl_repr = global_mean_pool(x, g.batch)
		else:
			raise Exception("Unknown graph_output_type")

		graph_lvl_output = None
		if self.graph_output_proj is not None:
			graph_lvl_output = self.graph_output_proj(graph_lvl_repr)

		return node_lvl_output, edge_lvl_output, graph_lvl_output
