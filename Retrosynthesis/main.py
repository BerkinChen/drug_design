from numpy import save
from torch.utils import data as torch_data
from torchdrug import core, models, tasks
import torch
from torchdrug import data, datasets, utils
from torchdrug.utils import plot

reaction_dataset = datasets.USPTO50k("../data/molecule-datasets/",
                                     node_feature="center_identification",
                                     kekulize=True)
synthon_dataset = datasets.USPTO50k("../data/molecule-datasets/", as_synthon=True,
                                    node_feature="synthon_completion",
                                    kekulize=True)

torch.manual_seed(1)
reaction_train, reaction_valid, reaction_test = reaction_dataset.split()
torch.manual_seed(1)
synthon_train, synthon_valid, synthon_test = synthon_dataset.split()


reaction_model = models.RGCN(input_dim=reaction_dataset.node_feature_dim,
                             hidden_dims=[256, 256, 256, 256, 256, 256],
                             num_relation=reaction_dataset.num_bond_type,
                             concat_hidden=True)
reaction_task = tasks.CenterIdentification(reaction_model,
                                           feature=("graph", "atom", "bond"))

synthon_model = models.RGCN(input_dim=synthon_dataset.node_feature_dim,
                            hidden_dims=[256, 256, 256, 256, 256, 256],
                            num_relation=synthon_dataset.num_bond_type,
                            concat_hidden=True)
synthon_task = tasks.SynthonCompletion(synthon_model, feature=("graph",))


reaction_task.preprocess(reaction_train, None, None)
synthon_task.preprocess(synthon_train, None, None)
task = tasks.Retrosynthesis(reaction_task, synthon_task, center_topk=2,
                            num_synthon_beam=5, max_prediction=10)


lengths = [len(reaction_valid) // 10,
           len(reaction_valid) - len(reaction_valid) // 10]
reaction_valid_small = torch_data.random_split(reaction_valid, lengths)[0]

optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)
solver = core.Engine(task, reaction_train, reaction_valid_small, reaction_test,
                     optimizer, gpus=[0], batch_size=32)


batch = []
reaction_set = set()
for sample in reaction_valid:
    if sample["reaction"] not in reaction_set:
        reaction_set.add(sample["reaction"])
        batch.append(sample)
        if len(batch) == 4:
            break
batch = data.graph_collate(batch)
batch = utils.cuda(batch)
predictions, num_prediction = task.predict(batch)

products = batch["graph"][1]
top1_index = num_prediction.cumsum(0) - num_prediction
for i in range(len(products)):
    reactant = predictions[top1_index[i]].connected_components()[0]
    product = products[i].connected_components()[0]
    plot.reaction(reactant, product,save_file='result'+str(i)+',png')
