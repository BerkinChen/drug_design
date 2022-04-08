from cmath import log
from torchdrug import data,metrics

with open('result.txt','r') as f:
    smiles_list = f.readlines()
smiles_list = list(set(smiles_list))
molecule = data.PackedMolecule.from_smiles(smiles_list)
logp = metrics.logP(molecule)
index = logp.argsort(descending=True)[:8]
mols = molecule[index]
logp = logp[index]
titles =[f'Molecule {i}\nlogP = {logp[i]:.4f}' for i in range(8)]
mols.visualize(titles,num_row=2,save_file='result_logP.png')

SA = metrics.SA(molecule)
index = SA.argsort(descending=True)[:8]
mols = molecule[index]
SA = SA[index]
titles =[f'Molecule {i}\nSA = {SA[i]:.4f}' for i in range(8)]
mols.visualize(titles,num_row=2,save_file='result_SA.png')

QED = metrics.QED(molecule)
index = QED.argsort(descending=True)[:8]
mols = molecule[index]
QED = QED[index]
titles =[f'Molecule {i}\nQED = {QED[i]:.4f}' for i in range(8)]
mols.visualize(titles,num_row=2,save_file='result_QED.png')