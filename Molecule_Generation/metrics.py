import moses
with open('result.txt','r') as f:
    smiles_list = f.readlines()
metrics = moses.get_all_metrics(smiles_list)
with open('metrics.txt','w') as f:
    for k,v in metrics.items():
        f.write(str((k,v)))
        f.write('\n')