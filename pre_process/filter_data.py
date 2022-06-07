'''
To filter out data points that are incompatible with the assumptions - 
1. Question len <= 15
2. Answers of len 1
3. Corresponding images of size 299x299x3

Resulting is a <split>.txt file containing the valid indices
example - python filter_data.py --split train\

TODO - Same pre-processing for val/test. Note they don't
have annotation data.
'''

import os
import json
import numpy as np
from tqdm import tqdm
import torch
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--split', default = 'train', help = 'specify split')
opt = parser.parse_args()

data_base_path = '../data/'
split = opt.split
q_json = os.path.join(data_base_path, split, 'questions.json')
a_json = os.path.join(data_base_path, split, 'annotations.json')

ques_data = json.load(open(q_json))
ann_data = json.load(open(a_json))
N = len(ques_data['questions'])
data_subtype = ques_data['data_subtype']

f_path = os.path.join(data_base_path, split, split + '.txt')
with open(f_path, 'w') as f:
    for i in tqdm(range(N)):
        q = ques_data['questions'][i]
        a = ann_data['annotations'][i]

        assert q['image_id'] == a['image_id']

        # Check Q&A validity
        if len(q['question'].split(' ')) > 15:
            continue # question longer than 15

        if len(a['multiple_choice_answer'].split(' ')) != 1:
            continue # only single word answers

        img_id = q['image_id']
        img_name = 'COCO_' + data_subtype + '_' \
             + (12-len(str(img_id)))*'0' + str(img_id) + '.pt' 

        img_name = os.path.join(data_base_path, split, 'img_features', img_name)
        img = torch.load(img_name) #cv2.imread(img_name)

        # Check img dim
        _,_,h,w = img.shape
        if h != 10 or w != 10:
            continue  

        f.write(str(i))
        f.write('\n')