# drug_design
The drug design project

## fix bug
- In torchdrug/metrics/rdkit/sascorer.py the function 'calculateScore' need add 'Chem.SanitizeMol(m)' before 'fp = rdMolDescriptors.GetMorganFingerprint(m, 2)'
