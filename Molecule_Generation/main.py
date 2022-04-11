from collections import defaultdict
from email.policy import default
import imp
from logging import exception
from pyrsistent import T
from torchdrug.layers import distribution
from torch import nn, optim
from torchdrug import core, models, tasks
import torch
from torchdrug import datasets
import os
import argparse

'''
    If you are synchronous distributed parallel training over multiple CPUs or GPUs, 
launch with one of the following commands.
    1. Single-node multi-process case.
        python -m torch.distributed.launch --nproc_per_node={number_of_gpus} {your_script.py} {your_arguments...}
    2. Multi-node multi-process case.
        python -m torch.distributed.launch --nnodes={number_of_nodes} --node_rank={rank_of_this_node}
        --nproc_per_node={number_of_gpus} {your_script.py} {your_arguments...}
'''

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='zinc_250k',
                    choices=['zinc_250k', 'zinc_2m', 'moses'], help='The dataset for training')
parser.add_argument('-o', '--output_file', type=str,
                    default='result.txt', help='The file to save the results')
parser.add_argument('-c', '--checkpoint', type=str,
                    default=None, help='The path to save the checkpoint')
parser.add_argument('-train_with_save', type=bool,
                    default=False, help='Save the checkpoint every epoch')
parser.add_argument('-gpus', type=list, default=None,
                    help='The list of gpus number')
parser.add_argument('-epoch', type=int, default=10,
                    help='The number of training epcohs')
parser.add_argument('-batch_size', type=int, default=128,
                    help='The batch size of training')
parser.add_argument('-log_interval', type=int, default=10,
                    help='The number of gradient updates times for log')
parser.add_argument('-aim', type=str, default=None,
                    choices=[None, 'logp', 'qed'], help='The aim to optimize')
parser.add_argument('-hidden_dims', type=list,
                    default=[256, 256, 256], help='The structure of hidden layers')
parser.add_argument('-num_layer', type=int, default=12,
                    help="The number of flow's layers")
parser.add_argument('-max_node', type=int, default=38,
                    help='The maximum number of nodes to generate molecules')
