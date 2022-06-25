from torchdrug import data, utils
import csv
from tqdm import tqdm

class Dataset(data.MoleculeDataset):
    target_fields = []

    def __init__(self, path, verbose=1, **kwargs):

        with open(path, "r") as fin:
            reader = csv.reader(fin)
            if verbose:
                reader = iter(tqdm(reader, "Loading %s" %
                              path, utils.get_line_count(path)))
            smiles_list = []

            for idx, values in enumerate(reader):
                smiles = values[0]
                smiles_list.append(smiles)

        targets = {}
        self.load_smiles(smiles_list, targets, lazy=False,
                         verbose=verbose, **kwargs)
