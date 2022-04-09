from collections import defaultdict
from logging import exception
from torchdrug.layers import distribution
from torch import nn, optim
from torchdrug import core, models, tasks
import torch
from torchdrug import datasets
import os


dataset = datasets.ZINC250k("../data/molecule-datasets/", kekulize=True,
                            node_feature="symbol")

model = models.RGCN(input_dim=dataset.num_atom_type,
                    num_relation=dataset.num_bond_type,
                    hidden_dims=[256, 256, 128], batch_norm=True)

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
                                      max_node=48, max_edge_unroll=12,
                                      criterion="nll")

optimizer = optim.Adam(task.parameters(), lr=1e-3)
solver = core.Engine(task, dataset, None, None, optimizer,
                     gpus=(1,), batch_size=32, log_interval=10)

if os.path.exists("../checkpoint/zinc250k_graphaf_molecule_generation_new.pnt"):
    solver.load("../checkpoint/zinc250k_graphaf_molecule_generation_new.pnt")
else:
    solver.train(num_epoch=10)
    solver.save("../checkpoint/zinc250k_graphaf_molecule_generation_new.pnt")

results = task.generate(num_sample=100000)
with open('result_10k.txt','w') as f:
    for m in results.to_smiles():
        f.write(m)
        f.write('\n')
#results.visualize(num_row=4,save_file='result.png')
