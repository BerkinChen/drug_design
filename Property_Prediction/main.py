from torchdrug import core, models, tasks, utils
import torch
from torchdrug import data, datasets
import random,time


dataset = datasets.ClinTox("../data/molecule-datasets/",verbose=1)
lengths = [int(0.8 * len(dataset)), int(0.1 * len(dataset))]
lengths += [len(dataset) - sum(lengths)]
train_set, valid_set, test_set = torch.utils.data.random_split(
    dataset, lengths)

graphs = []
labels = []
random.seed(time.time())
for j in range(8):
    i = random.randint(0,len(dataset)-1)
    sample = dataset[i]
    graphs.append(sample.pop("graph"))
    label = ["%s: %d" % (k, v) for k, v in sample.items()]
    label = ", ".join(label)
    labels.append(label)
graph = data.Molecule.pack(graphs)
graph.visualize(labels, num_row=2, save_file="tmp.png")


model = models.GIN(input_dim=dataset.node_feature_dim,
                   hidden_dims=[256, 256, 256, 256],
                   short_cut=True, batch_norm=True, concat_hidden=True)
task = tasks.PropertyPrediction(model, task=dataset.tasks,
                                criterion="bce", metric=("auprc", "auroc"), verbose=1)

optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)
solver = core.Engine(task, train_set, valid_set,
                     test_set, optimizer, gpus=[0], batch_size=1024)
#solver.train(num_epoch=500)
#solver.save('../checkpoint/clintox_gin_property_prediction.pnt')
solver.load('../checkpoint/clintox_gin_property_prediction.pnt')
solver.evaluate("valid")


samples = []
categories = set()
for sample in test_set[:16]:
    category = tuple([v for k, v in sample.items() if k != "graph"])
    categories.add(category)
    samples.append(sample)
samples = data.graph_collate(samples)
samples = utils.cuda(samples)

preds = torch.sigmoid(task.predict(samples))
targets = task.target(samples)

titles = []
for pred, target in zip(preds, targets):
    pred = ", ".join(["%.2f" % p for p in pred])
    target = ", ".join(["%d" % t for t in target])
    titles.append("predict: %s\ntarget: %s" % (pred, target))
graph = samples["graph"]
graph.visualize(titles, num_row=4, save_file='result.png')