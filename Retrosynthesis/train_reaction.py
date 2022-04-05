from torch.utils import data as torch_data
from torchdrug import core, models, tasks
import torch
from torchdrug.utils import plot
from torchdrug import data, datasets, utils

reaction_dataset = datasets.USPTO50k("../data/molecule-datasets/",
                                     node_feature="center_identification",
                                     kekulize=True)

for i in range(2):
    sample = reaction_dataset[i]
    reactant, product = sample["graph"]
    reactants = reactant.connected_components()[0]
    products = product.connected_components()[0]
    plot.reaction(reactants, products, save_file='tmp_reaction'+str(i)+'.png')


reaction_train, reaction_valid, reaction_test = reaction_dataset.split()

reaction_model = models.RGCN(input_dim=reaction_dataset.node_feature_dim,
                             hidden_dims=[256, 256, 256, 256, 256, 256],
                             num_relation=reaction_dataset.num_bond_type,
                             concat_hidden=True)
reaction_task = tasks.CenterIdentification(reaction_model,
                                           feature=("graph", "atom", "bond"))
reaction_optimizer = torch.optim.Adam(reaction_task.parameters(), lr=1e-3)
reaction_solver = core.Engine(reaction_task, reaction_train, reaction_valid,
                              reaction_test, reaction_optimizer,
                              gpus=[0], batch_size=128)
#reaction_solver.train(num_epoch=50)
reaction_solver.load("../checkpoint/uspto50k_g2gs_reactionl.pnt")
reaction_solver.evaluate("valid")
#reaction_solver.save("../checkpoint/uspto50k_g2gs_reactionl.pnt")

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
result = reaction_task.predict_synthon(batch)


def atoms_and_bonds(molecule, reaction_center):
    is_reaction_atom = (molecule.atom_map > 0) & \
                       (molecule.atom_map.unsqueeze(-1) ==
                        reaction_center.unsqueeze(0)).any(dim=-1)
    node_in, node_out = molecule.edge_list.t()[:2]
    edge_map = molecule.atom_map[molecule.edge_list[:, :2]]
    is_reaction_bond = (edge_map > 0).all(dim=-1) & \
                       (edge_map == reaction_center.unsqueeze(0)).all(dim=-1)
    atoms = is_reaction_atom.nonzero().flatten().tolist()
    bonds = is_reaction_bond[node_in < node_out].nonzero().flatten().tolist()
    return atoms, bonds


products = batch["graph"][1]
reaction_centers = result["reaction_center"]

for i, product in enumerate(products):
    true_atoms, true_bonds = atoms_and_bonds(product, product.reaction_center)
    true_atoms, true_bonds = set(true_atoms), set(true_bonds)
    pred_atoms, pred_bonds = atoms_and_bonds(product, reaction_centers[i])
    pred_atoms, pred_bonds = set(pred_atoms), set(pred_bonds)
    overlap_atoms = true_atoms.intersection(pred_atoms)
    overlap_bonds = true_bonds.intersection(pred_bonds)
    atoms = true_atoms.union(pred_atoms)
    bonds = true_bonds.union(pred_bonds)

    red = (1, 0.5, 0.5)
    blue = (0.5, 0.5, 1)
    purple = (1, 0.5, 1)
    atom_colors = {}
    bond_colors = {}
    for atom in atoms:
        if atom in overlap_atoms:
            atom_colors[atom] = purple
        elif atom in pred_atoms:
            atom_colors[atom] = red
        else:
            atom_colors[atom] = blue
    for bond in bonds:
        if bond in overlap_bonds:
            bond_colors[bond] = purple
        elif bond in pred_bonds:
            bond_colors[bond] = red
        else:
            bond_colors[bond] = blue

    plot.highlight(product, atoms, bonds, atom_colors, bond_colors,
                   save_file='tmp_reaction_center'+str(i)+'.png')

