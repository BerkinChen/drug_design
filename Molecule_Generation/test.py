import data
from torchdrug.layers import distribution
from torch import nn, optim
from torchdrug import core, models, tasks
import torch
from torchdrug import datasets
import os

dataset = data.Dataset("smiles.csv", kekulize=True,
                            node_feature="symbol")

model = models.RGCN(input_dim=dataset.num_atom_type,
                    num_relation=3,
                    hidden_dims=[256, 256, 256], batch_norm=True)
num_atom_type = dataset.num_atom_type
# add one class for non-edge
num_bond_type = 3 + 1

node_prior = distribution.IndependentGaussian(torch.zeros(num_atom_type),
                                              torch.ones(num_atom_type))
edge_prior = distribution.IndependentGaussian(torch.zeros(num_bond_type),
                                              torch.ones(num_bond_type))
node_flow = models.GraphAF(model, node_prior, num_layer=12)
edge_flow = models.GraphAF(model, edge_prior, use_edge=True, num_layer=12)

task = tasks.AutoregressiveGeneration(node_flow, edge_flow,
                                      max_node=36, max_edge_unroll=12,
                                      criterion="nll")

optimizer = optim.Adam(task.parameters(), lr=1e-3)
solver = core.Engine(task, dataset, None, None, optimizer,
                     gpus=(1,), batch_size=2, log_interval=10)

#solver.load("../checkpoint/zinc250k_graphaf_molecule_generation.pnt")
solver.train(num_epoch=200)

results = task.generate(num_sample=1000)
with open('result_owndata.txt', 'w') as f:
    for m in results.to_smiles():
        f.write(m)
        f.write('\n')
