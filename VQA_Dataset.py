'''
VQA_Data - Custom dataset class for the VAQ dataset.

Return a dict containing following keys - 
'quest_s' - List containing questions
'quest_e' - Bx1x15
'quest_l' - Tensor of original ques len
'ans_e'   - Bx1 
'ans_s',  - List containing anwers in string
'img_e'   - Bx2048x9x9


Example - 

dObj = VQA_Data('train')
train_dataloader = DataLoader(dObj, batch_size= 32, shuffle= False)
d = next(iter(train_dataloader))

'''
import os 
import os.path as osp
import torch
from torch.utils.data import Dataset
import numpy as np
import json

class VQA_Data(Dataset):
    def __init__(self, split, vocab_size = 3000):
        '''
        split - string ['train', 'val', 'test']
        '''
        self.split = split
        self.root = os.path.join('data', self.split)
        
        # Load questions
        q_json = json.load(open(osp.join(self.root, 'questions.json')))
        self.data_subtype = q_json['data_subtype']
        self.ques = q_json['questions']
        
        # Load answers
        self.ann = json.load(open(osp.join(self.root, 'annotations.json')))['annotations']       
        
        # Load dictionary for q&a
        if vocab_size == 3000:
            vocab = json.load(open('data/vocabulary_alternate.json'))
        else:
            vocab = json.load(open('data/vocabulary_1000.json'))
        self.q_vocab = vocab['questions']
        self.a_vocab = vocab['answers']
        
        # store size of ques & answer vocab
        self.a_vocab_size = len(self.a_vocab)
        self.q_vocab_size = len(self.q_vocab)
        
        self.index = open(os.path.join(self.root, self.split+'.txt')).readlines()
        
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        '''
        Returns a dict containing keys-
        'image_feature'
        'question enocded as per quest vocab'
        'answer encoded as per ans vocab'
        '''
        # map idx to idx in index
        idx = int(self.index[idx])
        data = {}
        # Question as string
        data['quest_s'] = self.ques[idx]['question']
        # Question encoded as ques vocab
        data['quest_e'], data['quest_l'] = self.word2vec(self.ques[idx]['question'][:-1])
        
        # Answer encoded as ans vocab
        data['ans_e'] = self.a_vocab.get(self.ann[idx]['multiple_choice_answer'], self.a_vocab_size)
        # Answer as string
        data['ans_s'] = self.ann[idx]['multiple_choice_answer']         
        data['ans_e'] = np.array([[data['ans_e']]])
        
        # Get image feature tensor.
        tensor_name = 'COCO_' + self.data_subtype + '_' \
         + (12-len(str(self.ques[idx]['image_id'])))*'0' + str(self.ques[idx]['image_id']) + '.pt' 
        data['img_e'] = torch.load(osp.join(self.root, 'img_features',  tensor_name))
        
        # Get image path
        img_name = 'COCO_' + self.data_subtype + '_' \
         + (12-len(str(self.ques[idx]['image_id'])))*'0' + str(self.ques[idx]['image_id']) + '.jpg' 
        data['img_p'] = osp.join(self.root, self.split+'2014',  img_name)
        
        return data
    
    def word2vec(self, text):
        '''
        Converts a sentence (array of string) to 
        an array of numbers based on vocab
        '''
        text = text.lower()
        enc = [self.q_vocab.get(word, self.q_vocab_size) for word in text.split(' ')]
        enc = np.array(enc)
        l = enc.shape[0]
        enc = np.pad(enc, (0, 15 - enc.shape[0])).reshape(1,-1)
        return enc, l