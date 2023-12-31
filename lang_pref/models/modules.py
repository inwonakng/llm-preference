'''
Copied over from https://github.com/zyli93/SAECON/blob/main/src/module.py
Citation: 
@inproceedings{saecon,
  author    = {Zeyu Li and
               Yilong Qin and
               Zihan Liu and
               Wei Wang},
  title     = {Powering Comparative Classification with Sentiment Analysis via Domain Adaptive Knowledge Transfer},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural
               Language Processing, 
               {EMNLP} 2021, 
               7-11 November 2021, 
               Online and in the Barceló Bávaro Convention Centre, Punta Cana, Dominican Republic},
  volume    = {{EMNLP} 2021},
  publisher = {Association for Computational Linguistics},
  year      = {2021},
}
'''

import torch
from torch import nn
from torch_geometric.nn import MessagePassing

class SGCNDir(MessagePassing):
    def __init__(
        self, 
        dim_in: int, 
        dim_out: int,
        num_labels: int,
        gating: bool
        ):
        super().__init__(aggr='add')
        self.W_dir = nn.Parameter(torch.FloatTensor(dim_in, dim_out).uniform_(-1,1))
        self.b_lab = nn.Parameter(torch.FloatTensor(num_labels, dim_out).uniform_(-1,1))
        self.gating = gating

        if self.gating:
            self.W_dir_g = nn.Parameter(torch.FloatTensor(dim_in, 1).uniform_(-1,1))
            self.b_lab_g = nn.Parameter(torch.FloatTensor(num_labels, 1).uniform_(-1,1))

    def forward(
        self, 
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_label: torch.LongTensor
        ) -> torch.Tensor:
        # x has shape [N, dim_in]
        # edge_index has shape [2, E]
        # edge_label has shape [E]

        b_lab = torch.index_select(self.b_lab, 0, edge_label)
        b_lab_g = torch.index_select(self.b_lab_g, 0, edge_label) if self.gating else None
    
        return self.propagate(edge_index, x=x, b_lab=b_lab, b_lab_g=b_lab_g)

    def message(
        self, 
        x_j: torch.Tensor, 
        b_lab: torch.Tensor, 
        b_lab_g: torch.Tensor
        ) -> torch.Tensor:
        # x_j has shape [E, dim_in]
        # b_lab has shape [E, dim_out]
        # b_lab_g has shape [E, 1]

        x_out = torch.matmul(x_j, self.W_dir) + b_lab

        if self.gating: 
            gate = torch.sigmoid(torch.matmul(x_j, self.W_dir_g) + b_lab_g)
            x_out = gate * x_out
        
        return x_out

class SGCNLoop(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        gating: bool
        ):
        super().__init__()
        self.lin = nn.Linear(dim_in, dim_out)
        self.gating = gating
        
        if self.gating:
            self.lin_g = nn.Linear(dim_in, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = self.lin(x)

        if self.gating:
            gate = torch.sigmoid(self.lin_g(x))
            x_out = gate * x_out
        
        return x_out
        
class SGCNConv(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_labels: int,
        gating: bool,
        directed: bool
        ):
        super().__init__()
        self.conv_loop = SGCNLoop(dim_in, dim_out, gating)
        self.directed = directed
        
        if self.directed:
            self.conv_in = SGCNDir(dim_in, dim_out, num_labels, gating)
            self.conv_out = SGCNDir(dim_in, dim_out, num_labels, gating)
        else:
            self.conv_dir = SGCNDir(dim_in, dim_out, num_labels, gating)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_label: torch.LongTensor
        ) -> torch.Tensor:
        x_loop = self.conv_loop(x)

        if self.directed:
            x_in = self.conv_in(x, edge_index, edge_label)
            x_out = self.conv_out(x, torch.flip(edge_index, (-2, )), edge_label)
            return torch.relu(x_loop + x_in + x_out)
        else:
            edge_index = torch.cat((edge_index, torch.flip(edge_index, (-2, ))), 1)
            edge_label = torch.cat((edge_label, edge_label))
            x_dir = self.conv_dir(x, edge_index, edge_label)
            return torch.relu(x_loop + x_dir)
