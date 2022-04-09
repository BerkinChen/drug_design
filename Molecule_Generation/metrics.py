import moses
import argparse

parser =  argparse.ArgumentParser()
parser.add_argument('-i','--inuput_file',dest='input_path',help='Please give the input file path')
parser.add_argument('-o','--output_file',dest='output_path',help='Please give the output file path')
args = parser.parse_args()



with open(args.input_path,'r') as f:
    smiles_list = f.readlines()
metrics = moses.get_all_metrics(smiles_list)
with open(args.output_path,'w') as f:
    for k,v in metrics.items():
        f.write(str((k,v)))
        f.write('\n')