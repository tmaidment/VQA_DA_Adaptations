import numpy as np
import glob as glob
import json
import argsparse
import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

## VQA Loading
parser = argparse.ArgumentParser(description='Preprocess VQA dataset to intermediate filetype')
parser.add_argument('--json', type=str, help='path to json file for questions')
parser.add_argument('--gpu', default=0, type=int, help='gpu to use on (default=0)')
args = parser.parse_args()

path_vqa_train = './datasets/qa/vqa/' 
path_vqa_test = 
