from collections import defaultdict
import imp
from logging import exception
from torchdrug.layers import distribution
from torch import nn, optim
from torchdrug import core, models, tasks
import torch
from torchdrug import datasets
import os


dataset = datasets.ZINC2m("../data/molecule-datasets/", kekulize=True,
                            node_feature="symbol")

model = models.RGCN(input_dim=dataset.num_atom_type,
                    num_relation=dataset.num_bond_type,
                    hidden_dims=[256, 256, 256], batch_norm=True)

num_atom_type = dataset.num_atom_type
# add one class for non-edge
num_bond_type = dataset.num_bond_type + 1

node_prior = distribution.IndependentGaussian(torch.zeros(num_atom_type),
                                              torch.ones(num_atom_type))
edge_prior = distribution.IndependentGaussian(torch.zeros(num_bond_type),
                                              torch.ones(num_bond_type))
node_flow = models.GraphAF(model, node_prior, num_layer=12)
edge_flow = models.GraphAF(model, edge_prior, use_edge=True, num_layer=12)

task = tasks.AutoregressiveGeneration(node_flow, edge_flow,
                                      max_node=38, max_edge_unroll=12,
                                      criterion="nll")

optimizer = optim.Adam(task.parameters(), lr=1e-3)
solver = core.Engine(task, dataset, None, None, optimizer,
            gpus=(0,), batch_size=128, log_interval=10)

if os.path.exists("../checkpoint/zinc2m_graphaf_molecule_generation.pnt"):
    solver.load("../checkpoint/zinc2m_graphaf_molecule_generation.pnt")
else:
    solver.train(num_epoch=10)
    solver.save("../checkpoint/zinc2m_graphaf_molecule_generation.pnt")

results = task.generate(num_sample=300000)
with open('result_2m.txt', 'w') as f:
    for m in results.to_smiles():
        f.write(m)
        f.write('\n')
# results.visualize(num_row=4,save_file='result.png')
