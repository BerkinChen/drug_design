import torch
from torch import nn, optim
from torch.utils import data as torch_data

from torchdrug import core, datasets, tasks, models, data

dataset = datasets.BACE("../data/molecule-datasets/",
                        node_feature="pretrain", edge_feature="pretrain")
lengths = [int(0.8 * len(dataset)), int(0.1 * len(dataset))]
lengths += [len(dataset) - sum(lengths)]
train_set, valid_set, test_set = data.ordered_scaffold_split(dataset, lengths)

model = models.GIN(input_dim=dataset.node_feature_dim,
                   hidden_dims=[300, 300, 300, 300, 300],
                   edge_input_dim=dataset.edge_feature_dim,
                   batch_norm=True, readout="mean")
task = tasks.PropertyPrediction(model, task=dataset.tasks,
                                criterion="bce", metric=("auprc", "auroc"))

optimizer = optim.Adam(task.parameters(), lr=1e-3)
solver = core.Engine(task, train_set, valid_set, test_set, optimizer,
                     gpus=[0], batch_size=256)

checkpoint = torch.load("../checkpoint/clintox_gin_attributemasking.pth")["model"]
task.load_state_dict(checkpoint, strict=False)

solver.train(num_epoch=100)
solver.evaluate("valid")
