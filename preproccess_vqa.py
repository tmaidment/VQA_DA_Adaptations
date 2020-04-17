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

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
random.seed(0)

def get_image_id(path):
    return path.split('_')[-1][:-4]

def temp_question(question_str):
    return np.random.rand(4, 1000)

def temp_answer(answer_str):
    return np.random.rand(4, 1000)

class QuestionImages(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __getitem__(self, i):
        path = self.image_paths[i]
        img = Image.open(path)
        img = img.convert('RGB')
        return (get_image_id(path), self.transform(img))

    def __len__(self):
        return len(self.image_paths)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess VQA dataset to intermediate filetype')
    parser.add_argument('--q_json', type=str, help='path to json file for questions')
    parser.add_argument('--a_json', type=str, help='path to json file for questions')
    parser.add_argument('--gpu', default=0, type=int, help='gpu to use on (default=0)')
    parser.add_argument('--coco_folder', default='./datasets/imgs', help='root dir for all coco images')
    parser.add_argument('--output_folder', default='./preproccessed/', help='the folder to output processed images to')
    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    #device = torch.device('cpu')
    cpu_device = torch.device('cpu')

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

    coco_images = glob.glob('{}/*.jpg'.format(os.path.join(args.coco_folder,task)))
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = QuestionImages(coco_images[:10], transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)


    model = torchvision.models.resnet50(pretrained=True)
    model.to(device)
    model.eval()
    
    image_features = {}
    print('Extracting image features...')
    t = tqdm(iter(dataloader), total=len(dataloader))
    for image_ids, images in t:
        images = images.to(device)
        outputs = model(images)
        for idx, image_id in enumerate(image_ids):
            image_features[image_id] = outputs[idx].detach().cpu().numpy()

    mc_ans = {}
    print('Extracting answers...')
    for ans in tqdm(json_as):
        #if ans['image_id'] in image_features:
        mc_ans[ans['question_id']] = ans['multiple_choice_answer']

    print('Saving processed files...')
    for qs in tqdm(json_qs):
        image_id = str(qs['image_id']).zfill(12)
        if image_id in image_features:
            qid = qs['question_id']
            answer = mc_ans[qid]
            choices = qs['multiple_choices']
            choices.remove(answer)
            answers = random.choices(choices, k=3)
            answers.append(answer)
            random.shuffle(answers)
            answer_idx = answers.index(answer)

            i_vec = image_features[image_id]
            i_vec -= i_vec.min()
            i_vec /= i_vec.max()
            q_vec = temp_question(qs['question'])
            a_vec = temp_answer(answers)
            
            combined = np.vstack((q_vec, a_vec, np.expand_dims(i_vec, 0))) * 255
            img = combined.astype(np.uint8)

            im = Image.fromarray(combined)
            im = im.convert("L")
            path = os.path.join(args.output_folder,task,str(answer_idx))
            if not os.path.exists(path):
                os.makedirs(path)
            with open(os.path.join(path,'{}.png'.format(qid)), 'wb') as f:
                im.save(f)
            # imwrite(os.path.join(path,'{}.png'.format(qas['question_id'])), img)
            #print(qas)
    

