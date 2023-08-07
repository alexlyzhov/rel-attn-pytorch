import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class NodeRelAttn(torch.nn.Module):
	def __init__(self, node_dim, edge_dim, num_heads, weights_dropout_p):
		super().__init__()
		self.num_heads = num_heads
		self.node_dim = node_dim
		self.edge_dim = edge_dim
		self.head_dim = node_dim // num_heads

		self.scale = 1.0 / np.sqrt(self.head_dim)

		self.nodes_q_transform = nn.Linear(node_dim, node_dim)
		self.nodes_k_transform = nn.Linear(node_dim, node_dim)
		self.nodes_v_transform = nn.Linear(node_dim, node_dim)

		self.edges_q_transform = nn.Linear(edge_dim, node_dim)
		self.edges_k_transform = nn.Linear(edge_dim, node_dim)
		self.edges_v_transform = nn.Linear(edge_dim, node_dim)

		self.weights_dropout = nn.Dropout(weights_dropout_p)

	def forward(self, node_tensors, edge_tensors):
		# node_tensors.shape == (batch_size, num_nodes, node_dim)
		num_nodes = node_tensors.shape[1]

		nodes_q = self.nodes_q_transform(node_tensors)
		nodes_k = self.nodes_k_transform(node_tensors)
		nodes_v = self.nodes_v_transform(node_tensors)

		edges_q = self.edges_q_transform(edge_tensors)
		edges_k = self.edges_k_transform(edge_tensors)
		edges_v = self.edges_v_transform(edge_tensors)

		nodes_q = self.separate_node_heads(nodes_q)
		nodes_k = self.separate_node_heads(nodes_k)
		nodes_v = self.separate_node_heads(nodes_v)

		edges_q = self.separate_edge_heads(edges_q)
		edges_k = self.separate_edge_heads(edges_k)
		edges_v = self.separate_edge_heads(edges_v)

		q = edges_q + nodes_q.reshape((-1, self.num_heads, num_nodes, 1, self.head_dim))  # -1 is batch_size everywhere
		k = edges_k + nodes_k.reshape((-1, self.num_heads, 1, num_nodes, self.head_dim))
		q = q.reshape((-1, self.num_heads, num_nodes, num_nodes, 1, self.head_dim))
		k = k.reshape((-1, self.num_heads, num_nodes, num_nodes, self.head_dim, 1))
		qk = torch.matmul(q, k)
		qk = qk.reshape((-1, self.num_heads, num_nodes, num_nodes))

		qk = qk * self.scale

		att_dist = torch.nn.functional.softmax(qk, dim=-1)
		att_dist = self.weights_dropout(att_dist)

		att_dist = att_dist.reshape((-1, self.num_heads, num_nodes, 1, num_nodes))
		v2 = edges_v + nodes_v.reshape((-1, self.num_heads, 1, num_nodes, self.head_dim))
		new_nodes = torch.matmul(att_dist, v2)
		new_nodes = new_nodes.reshape((-1, self.num_heads, num_nodes, self.head_dim))

		res = self.concatenate_heads(new_nodes)
		return res

	def separate_node_heads(self, x):
		new_shape = x.shape[:-1] + (self.num_heads, self.head_dim)
		x = x.reshape(new_shape)
		x = x.permute((0, 2, 1, 3))  # (batch_size, num_heads, num_nodes, head_dim)
		return x

	def separate_edge_heads(self, x):
		new_shape = x.shape[:-1] + (self.num_heads, self.head_dim)
		x = x.reshape(new_shape)
		x = x.permute((0, 3, 1, 2, 4))  # (Batch, Heads, Nodes, Nodes, Head size)
		return x
	
	def concatenate_heads(self, x):
		x = x.permute((0, 2, 1, 3))
		new_shape = x.shape[:-2] + (self.node_dim,)
		return x.reshape(new_shape)


class RelAttnLayer(nn.Module):
	def __init__(
		self,
		node_dim: int,
		edge_dim: int,
		num_heads: int,
		NHS: int,
		EHS1: int,
		EHS2: int,
		dropout_rate: float,
		v2_flag: bool,
		weights_dropout_p: float,
	):
		super().__init__()
		self.node_dim = node_dim
		self.edge_dim = edge_dim
		self.num_heads = num_heads
		self.head_dim = self.node_dim // self.num_heads
		self.NHS = NHS
		self.EHS1 = EHS1
		self.EHS2 = EHS2
		self.dropout_rate = dropout_rate
		self.v2_flag = v2_flag
		self.weights_dropout_p = weights_dropout_p

		self.node_attn = NodeRelAttn(self.node_dim, self.edge_dim, self.num_heads, self.weights_dropout_p)
		self.node_lin_1 = nn.Linear(self.node_dim, self.node_dim)
		self.node_layer_norm_1 = nn.LayerNorm(self.node_dim)
		self.node_lin_2 = nn.Linear(self.node_dim, self.NHS)
		self.node_lin_3 = nn.Linear(self.NHS, self.node_dim)
		self.node_layer_norm_2 = nn.LayerNorm(self.node_dim)

		if self.v2_flag:
			edge_lin_1_dim = self.edge_dim * 2 + self.node_dim * 4
		else:
			edge_lin_1_dim = self.edge_dim * 2 + self.node_dim * 2
		self.edge_lin_1 = nn.Linear(edge_lin_1_dim, self.EHS1)
		self.edge_lin_2 = nn.Linear(self.EHS1, self.edge_dim)
		self.edge_layer_norm_1 = nn.LayerNorm(self.edge_dim)

		self.edge_lin_3 = nn.Linear(self.edge_dim, self.EHS2)
		self.edge_lin_4 = nn.Linear(self.EHS2, self.edge_dim)
		self.edge_layer_norm_2 = nn.LayerNorm(self.edge_dim)

		self.dropout = nn.Dropout(self.dropout_rate)

	def forward(self, node_tensors, edge_tensors):
		num_nodes = node_tensors.shape[1]

		attw_node_tensors = self.node_attn(node_tensors, edge_tensors)
		residuals = self.dropout(self.node_lin_1(attw_node_tensors))
		node_tensors = self.node_layer_norm_1(node_tensors + residuals)

		residuals = self.dropout(self.node_lin_3(F.relu(self.node_lin_2(node_tensors))))
		node_tensors = self.node_layer_norm_2(node_tensors + residuals)

		source_nodes = node_tensors.unsqueeze(1)
		# (batch_size, 1, num_nodes, node_dim)
		expanded_source_nodes = source_nodes.repeat(1, num_nodes, 1, 1)
		# (batch_size, num_nodes, num_nodes, node_dim)
		target_nodes = node_tensors.unsqueeze(2)
		expanded_target_nodes = target_nodes.repeat(1, 1, num_nodes, 1)

		reversed_edge_tensors = edge_tensors.transpose(-2, -3)
		input_tensors = (edge_tensors, reversed_edge_tensors, expanded_source_nodes, expanded_target_nodes)

		if self.v2_flag:
			attw_source_nodes = attw_node_tensors.unsqueeze(1)
			expanded_attw_source_nodes = attw_source_nodes.repeat(1, num_nodes, 1, 1)
			attw_target_nodes = attw_node_tensors.unsqueeze(2)
			expanded_attw_target_nodes = attw_target_nodes.repeat(1, 1, num_nodes, 1)

			input_tensors += (expanded_attw_source_nodes, expanded_attw_target_nodes,)

		concatenated_inputs = torch.cat(input_tensors, dim=-1)

		residuals = self.dropout(self.edge_lin_2(F.relu(self.edge_lin_1(concatenated_inputs))))
		edge_tensors = self.edge_layer_norm_1(edge_tensors + residuals)

		residuals = self.dropout(self.edge_lin_4(F.relu(self.edge_lin_3(edge_tensors))))
		edge_tensors = self.edge_layer_norm_2(edge_tensors + residuals)

		return node_tensors, edge_tensors
