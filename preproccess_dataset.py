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
from helper_functions import feature_vectors, Q_interesting_words

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
#random.seed(0)

def get_image_id(path):
    return path.split('_')[-1][:-4]

def normalize(vec):
    vec -= vec.min()
    vec /= vec.max()
    return vec

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
    parser.add_argument('--dataset', type=str, help='folder containing the text files')
    parser.add_argument('--gpu', default=0, type=int, help='gpu to use on (default=0)')
    parser.add_argument('--coco_folder', default='./datasets/imgs/', help='root dir for all coco images')
    parser.add_argument('--output_folder', help='the folder to output processed images to')
    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    #device = torch.device('cpu')
    cpu_device = torch.device('cpu')

    print('Loading dataset...')
    with open(os.path.join(args.dataset, 'questions.txt')) as f:
        questions = f.readlines()

    with open(os.path.join(args.dataset, 'answers.txt')) as f:
        answers = f.readlines()

    with open(os.path.join(args.dataset, 'img_ids.txt')) as f:
        img_ids = f.readlines()

    if 'train' in args.dataset:
        task = 'train2014'
    else:
        task = 'val2014'

    coco_images = glob.glob('{}/*.jpg'.format(os.path.join(args.coco_folder,task)))
    #coco_images = coco_images[:10]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = QuestionImages(coco_images, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

    model = torchvision.models.resnet50(pretrained=True)
    model.to(device)
    model.eval()
    
    image_features = {}
    print('Extracting image features...')
    t = tqdm(iter(dataloader), leave=False, total=len(dataloader))
    for image_ids, images in t:
        images = images.to(device)
        outputs = model(images)
        for idx, image_id in enumerate(image_ids):
            image_features[image_id] = outputs[idx].detach().cpu().numpy()
    print('Saving processed files...')
    for idx, (question, answer, image_id) in tqdm(enumerate(zip(questions, answers, img_ids)), total=len(questions)):
        image_id = image_id.strip().zfill(12)
        if image_id in image_features:
            try:
                candidate_answers = random.sample(answers, 4)
                if answer in candidate_answers:
                    candidate_answers.remove(answer)
                else:
                    candidate_answers = candidate_answers[:-1]
                candidate_answers.append(answer)
                random.shuffle(candidate_answers)
                answer_idx = candidate_answers.index(answer)
                
                q_words = Q_interesting_words.extract_four_words_from_question(question.strip())
                q_vecs = []
                a_vecs = []
                for q_word in q_words:
                    if q_word != ' ':
                        q_vecs.append(feature_vectors.question_vector(q_word.strip()))
                    else:
                        q_vecs.append(list(np.zeros(100)))
                for a_word in candidate_answers:
                    a_vecs.append(feature_vectors.answer_vector(a_word.strip()))

                q_vec = np.asarray(q_vecs)
                a_vec = np.asarray(a_vecs)
                i_vec = image_features[image_id].reshape(10, 100)

                combined = np.vstack((q_vec, a_vec, i_vec)) * 255
                img = combined.astype(np.uint8)

                im = Image.fromarray(combined)
                im = im.convert("L")

                path = os.path.join(args.output_folder,task,str(answer_idx))
                if not os.path.exists(path):
                    os.makedirs(path)
                with open(os.path.join(path,'{}.png'.format(idx)), 'wb') as f:
                    im.save(f)
            except:
                pass
            
            
            #stack everything here

            #print('stop')

        

