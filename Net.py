'''
This file creates the different types of models for
the VQA task
'''

import torch.nn as nn
import torch
from Model import TextProcessing, Classifier, Fusion, BaselineFusion, StackedAttention, average_layer


'''
Tiled attention model
Refer - https://arxiv.org/pdf/1704.03162.pdf
'''
class TiledAttention(nn.Module):
    # init method to initialize and the forward function
    def __init__(self, l_dict, dropout = None, num_classes = 3001):
        super(TiledAttention,self).__init__()
        glimpses=2
        img_vec_size=2048
        text_vec_size = 1024
        ques_len=300 # size of embedding vector
                
        self.text = TextProcessing(l_dict,l_ques=ques_len,embedding_size=text_vec_size, dropout = dropout)
        self.fuse = Fusion(v=img_vec_size,q=text_vec_size,mid=512,glimpses=glimpses, dropout = dropout)
        self.classifier =  Classifier(in_features=glimpses*img_vec_size+text_vec_size,mid_features=text_vec_size,out_features=num_classes, dropout = dropout)
        
    def forward(self,text_vec,img_vec,q_len):
        text_vec=self.text(text_vec,q_len) #text_processing
        f = self.fuse(img_vec,text_vec) #fuse_image_vec and text_vec
        img_vec = average_layer(img_vec,f) #pass through softmax
        vec = torch.cat([img_vec,text_vec], dim = 1) #concatenate text_proc with this heat map
        out = self.classifier(vec)#pass through classifier
        return out
    

'''
Baseline Model
Refer - https://arxiv.org/pdf/1505.00468v6.pdf
'''
class BaseLine(nn.Module):
    def __init__(self, l_dict, dropout = None, num_classes = 3001):
        super(BaseLine,self).__init__()
        text_vec_size = 1024
        ques_len=300 # size of embedding vector
        
        self.text = TextProcessing(l_dict,l_ques=ques_len,embedding_size=text_vec_size, dropout = dropout)
        self.fusion = BaselineFusion(num_classes)
    
    def forward(self,text_vec,img_vec,q_len):
        text_vec=self.text(text_vec,q_len) #text_processing
        pred = self.fusion(img_vec, text_vec)
        return pred
    
'''
Stacked attention Model
Refer - https://arxiv.org/pdf/1511.02274.pdf
'''  
class SANModel(nn.Module):
    def __init__(self, l_dict, dropout = None, num_classes = 3001):
        super(SANModel,self).__init__()
        text_vec_size = 1024
        ques_len=300 # size of embedding vector
        num_classes = num_classes
                
        self.text = TextProcessing(l_dict,l_ques=ques_len,embedding_size=text_vec_size, dropout = dropout)
        self.fusion = StackedAttention(num_classes)
    
    def forward(self,text_vec,img_vec,q_len):
        text_vec=self.text(text_vec,q_len) #text_processing
        pred = self.fusion(img_vec, text_vec)
        return pred