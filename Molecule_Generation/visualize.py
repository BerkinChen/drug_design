from pyrsistent import T
from torchdrug import data, metrics
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('-i', '--input_file', dest='input_path',
                   default=None, help='The input file path', required=True)
parse.add_argument('-o', '--output_file', dest='output_path', default='result',
                   help='The output file name, such as result')
parse.add_argument('-n', '--num', dest='num', default=8,
                   type=int, help='The number of Molecules')
args = parse.parse_args()

with open(args.input_path, 'r') as f:
    smiles_list = f.readlines()
smiles_list = list(set(smiles_list))
molecule = data.PackedMolecule.from_smiles(smiles_list)
logp = metrics.logP(molecule)
index = logp.argsort(descending=True)[:args.num]
mols = molecule[index]
logp = logp[index]
titles = [f'Molecule {i+1}\nlogP = {logp[i]:.4f}' for i in range(args.num)]
mols.visualize(titles, num_row=args.num//4,
               save_file=args.output_path+'_logP.png')

SA = metrics.SA(molecule)
index = SA.argsort(descending=True)[:args.num]
mols = molecule[index]
SA = SA[index]
titles = [f'Molecule {i+1}\nSA = {SA[i]:.4f}' for i in range(args.num)]
mols.visualize(titles, num_row=args.num//4,
               save_file=args.output_path+'_SA.png')

QED = metrics.QED(molecule)
index = QED.argsort(descending=True)[:args.num]
mols = molecule[index]
QED = QED[index]
titles = [f'Molecule {i+1}\nQED = {QED[i]:.4f}' for i in range(args.num)]
mols.visualize(titles, num_row=args.num//4,
               save_file=args.output_path+'_QED.png')
