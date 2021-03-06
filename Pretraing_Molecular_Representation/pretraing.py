import torch
from torch import nn, optim
from torch.utils import data as torch_data
import os
from torchdrug import core, datasets, tasks, models, data

dataset = datasets.ClinTox("../data/molecule-datasets/", node_feature="pretrain",
                           edge_feature="pretrain")

model = models.GIN(input_dim=dataset.node_feature_dim,
                   hidden_dims=[300, 300, 300, 300, 300],
                   edge_input_dim=dataset.edge_feature_dim,
                   batch_norm=True, readout="mean")
task = tasks.AttributeMasking(model, mask_rate=0.15)

optimizer = optim.Adam(task.parameters(), lr=1e-3)
solver = core.Engine(task, dataset, None, None,
                     optimizer, gpus=[0], batch_size=256)

if not os.path.exists("../checkpoint/clintox_gin_attributemasking.pnt"):
    solver.train(num_epoch=100)
    solver.save("../checkpoint/clintox_gin_attributemasking.pnt")