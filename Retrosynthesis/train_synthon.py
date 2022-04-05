from torch.utils import data as torch_data
from torchdrug import core, models, tasks
import torch
from torchdrug.utils import plot
from torchdrug import data, datasets, utils

synthon_dataset = datasets.USPTO50k("../data/molecule-datasets/", as_synthon=True,
                                    node_feature="synthon_completion",
                                    kekulize=True)

for i in range(3):
    sample = synthon_dataset[i]
    reactant, synthon = sample["graph"]
    plot.reaction([reactant], [synthon], save_file='tmp_synthon'+str(i)+'.png')

synthon_train, synthon_valid, synthon_test = synthon_dataset.split()

synthon_model = models.RGCN(input_dim=synthon_dataset.node_feature_dim,
                            hidden_dims=[256, 256, 256, 256, 256, 256],
                            num_relation=synthon_dataset.num_bond_type,
                            concat_hidden=True)
synthon_task = tasks.SynthonCompletion(synthon_model, feature=("graph",))

synthon_optimizer = torch.optim.Adam(synthon_task.parameters(), lr=1e-3)
synthon_solver = core.Engine(synthon_task, synthon_train, synthon_valid,
                             synthon_test, synthon_optimizer,
                             gpus=[1], batch_size=128)
synthon_solver.train(num_epoch=10)
synthon_solver.evaluate("valid")
synthon_solver.save("../checkpoint/uspto50k_g2gs_syntho.pnt")

batch = []
reaction_set = set()
for sample in synthon_valid:
    if sample["reaction"] not in reaction_set:
        reaction_set.add(sample["reaction"])
        batch.append(sample)
        if len(batch) == 4:
            break
batch = data.graph_collate(batch)
batch = utils.cuda(batch)
reactants, synthons = batch["graph"]
reactants = reactants.ion_to_molecule()
predictions = synthon_task.predict_reactant(
    batch, num_beam=10, max_prediction=5)

synthon_id = -1
i = 0
titles = []
graphs = []
for prediction in predictions:
    if synthon_id != prediction.synthon_id:
        synthon_id = prediction.synthon_id.item()
        i = 0
        graphs.append(reactants[synthon_id])
        titles.append("Truth %d" % synthon_id)
    i += 1
    graphs.append(prediction)
    if reactants[synthon_id] == prediction:
        titles.append("Prediction %d-%d, Correct!" % (synthon_id, i))
    else:
        titles.append("Prediction %d-%d" % (synthon_id, i))

# reset attributes so that pack can work properly
mols = [graph.to_molecule() for graph in graphs]
graphs = data.PackedMolecule.from_molecule(mols)
graphs.visualize(titles, save_file="uspto50k_synthon_valid.png", num_col=6)


