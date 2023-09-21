import torch
from torch_geometric.nn import GATConv,GCNConv,GATv2Conv,SAGEConv,TransformerConv
import torch.nn.functional as F
from sklearn.metrics import f1_score
import math
import numpy as np
from models.modules import SGCNConv
from models.pos_encoder import PositionalEncoding

GNN_MAPPINGS = {
    'GAT': GATConv,
    'GATv2': GATv2Conv,
    'Transformer': TransformerConv,
    'GCN': GCNConv,
    'SGCN': SGCNConv,
    'SAGE': SAGEConv,
}
HAS_ATTENTION = ['GAT','GATv2','Transformer']
NO_ATTENTION = ['GCN','SAGE']

class ED_GNN(torch.nn.Module):
    def __init__(self,
        layer_type:str = 'GAT',
        n_layers:int=8, 
        hidden_channels:int=300, 
        heads:int=6, 
        n_features:int=768, 
        use_edge_attr = False,
        use_pos_encoding = False,
        n_classes = 3,
        dropout=0.05
    ):
        self.use_edge_attr = use_edge_attr
        self.layer_type = layer_type
        self.dropout = dropout
        
        super(ED_GNN,self).__init__()
        self.start_layer = torch.nn.Linear(n_features, hidden_channels)
        
        self.use_pos_encoding = use_pos_encoding
        if self.use_pos_encoding:
            self.encoder = PositionalEncoding(n_features)
        
        if self.layer_type in HAS_ATTENTION:
            self.graph_layers = torch.nn.ModuleList([
                GNN_MAPPINGS[self.layer_type](
                    in_channels=hidden_channels,
                    out_channels=hidden_channels//heads, 
                    heads=heads,
                    dropout=dropout,
                    # this parameter is necessary for using edge attributes
                    # transformer does not follow starndard api... 
                    edge_dim = (0 if self.layer_type == 'Transformer' else 1) 
                                if use_edge_attr else 
                                None
                )
                for _ in range(n_layers)
            ])

        elif self.layer_type in NO_ATTENTION:
            self.graph_layers = torch.nn.ModuleList([
                GNN_MAPPINGS[self.layer_type](
                    in_channels=hidden_channels,
                    out_channels=hidden_channels, 
                )
                for _ in range(n_layers)
            ])

            self.use_edge_attr = False

        # corner case since this doesn't foollow standard API
        elif self.layer_type == 'SGCN':
            if not self.use_edge_attr: 
                print('SGCN requires edge attributes')
                return
            self.graph_layers = torch.nn.ModuleList([
                SGCNConv(
                    hidden_channels,
                    hidden_channels,
                    343, # very hardcoded. We actually only have around 50, but each label is represented by an index
                    True,
                    True,   

                )
                for _ in range(n_layers)
            ])

        self.final_layer = torch.nn.Linear(hidden_channels*2, n_classes)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        a_mask = data.a_mask
        b_mask = data.b_mask
        ptr = data.ptr
        
        if self.use_pos_encoding:
            x = self.encoder(x[None,:])[0]
        
        x = F.dropout(x,p=0.3,training=self.training) # I think this means word-embedding dropout of 0.3
        x = self.start_layer(x) # H0 = XW0 + b0
        
        # these guys can handle edge attr
        if self.layer_type in HAS_ATTENTION:
            for l in self.graph_layers:
                x = F.elu(
                    l(  x,
                        edge_index,
                        edge_attr = edge_attr if self.use_edge_attr else None
                    )
                )
        elif self.layer_type in NO_ATTENTION:
            for l in self.graph_layers:
                x = F.elu(
                    F.dropout(
                        l(  x,
                            edge_index
                        ),
                        p = self.dropout,
                        training=self.training
                    )
                )

        elif self.layer_type == 'SGCN':
            for l in self.graph_layers:
                x = F.elu(
                    F.dropout(
                        l(  x,
                            edge_index,
                            edge_attr.long()
                        ),
                        p = self.dropout,
                        training=self.training
                    )
                )
        
        x = torch.vstack([
            torch.concat([
                x[beg:end][a_mask[beg:end]].mean(0),
                x[beg:end][b_mask[beg:end]].mean(0)
            ])
            for beg,end in zip(ptr[:-1],ptr[1:])
        ]).nan_to_num()

        return F.softmax(self.final_layer(x),dim=1)

