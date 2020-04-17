import numpy as np
import glob as glob
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
import torchvision
import os
from PIL import Image
from tqdm import tqdm
import random
import re

np.random.seed(0)
random.seed(0)

def clean_string(string):
    return re.sub('\W+',' ', string).lower()

def get_dataset_split(string):
    return string.split('_')[-2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract questions from VQA')
    parser.add_argument('--q_json', type=str, help='path to json file for questions')
    parser.add_argument('--a_json', type=str, help='path to json file for questions')
    parser.add_argument('--output_folder', default='./datasets/qa/vqa', help='the folder to output to')
    args = parser.parse_args()

    with open(args.q_json) as f:
        json_raw = json.load(f)
        json_qs = json_raw['questions']
        task = json_raw['data_subtype']

    with open(args.a_json) as f:
        json_raw = json.load(f)
        json_as = json_raw['annotations']
        task_check = json_raw['data_subtype']
        if task_check != task:
            raise Exception('Q and A jsons don\'t match.')
    
    mc_ans = {}
    print('Loading answers...')
    for ans in tqdm(json_as):
        #if ans['image_id'] in image_features:
        mc_ans[ans['question_id']] = ans['multiple_choice_answer']

    print('Saving processed files...')
    data_split = get_dataset_split(args.q_json)
    path = os.path.join(args.output_folder,data_split)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path,'questions.txt'.format(task)), 'a') as q_f:    
        with open(os.path.join(path,'answers.txt'.format(task)), 'a') as a_f:
            with open(os.path.join(path,'img_ids.txt'.format(task)), 'a') as i_f:
                for qs in tqdm(json_qs):
                    #image_id = str(qs['image_id']).zfill(12)
                    qid = qs['question_id']
                    answer = mc_ans[qid]
                    # choices = qs['multiple_choices']
                    # choices.remove(answer)
                    # answers = random.choices(choices, k=3)
                    # answers.append(answer)
                    # random.shuffle(answers)
                    # answer_idx = answers.index(answer)
                    # question = clean_string(qs['question'])
                    # answers = ','.join(answers)
                    if len(answer.split(' ')) == 1:
                        i_f.write(str(qs['image_id'])+'\n')
                        q_f.write(clean_string(qs['question'])+'\n')
                        a_f.write(answer+'\n')
                




        # imwrite(os.path.join(path,'{}.png'.format(qas['question_id'])), img)
        #print(qas)
    

