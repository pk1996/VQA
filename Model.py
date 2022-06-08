'''
This file contains all the blocks needed to construct
different models
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.init as init



'''
LSTM based component to extract features 
from question.
'''
class TextProcessing(nn.Module):
    def __init__(self,l_dict,l_ques,embedding_size, dropout = None):
        super(TextProcessing, self).__init__()
        self.embedding=nn.Embedding(l_dict,l_ques,padding_idx=0)
        self.dropout = dropout
        if self.dropout:
            self.drop = nn.Dropout(self.dropout)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_size=l_ques,hidden_size=embedding_size)
        self._init_xavier_weights(self.lstm.weight_ih_l0)
        self._init_xavier_weights(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()
        init.xavier_uniform_(self.embedding.weight)
        
    def _init_xavier_weights(self,weights):
        init.xavier_uniform_(weights)
        
    def forward(self,question,l_ques):
        embed = self.embedding(question)
        if self.dropout:
            embed = self.drop(embed)
        th = self.tanh(embed)
        pack = pack_padded_sequence(th,l_ques,batch_first=True, enforce_sorted = False)
        _, (_, c) = self.lstm(pack)
        return c.squeeze(0)

'''
Classifier block used by tiled attention model
'''    
class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features=1024, out_features=3000, dropout = None):
        super(Classifier, self).__init__()
        drop = dropout
        if drop is not None:
            self.add_module('drop1', nn.Dropout(drop))
        self.add_module('1024fc', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        if drop is not None:
            self.add_module('drop2', nn.Dropout(drop))
        self.add_module('3000fc', nn.Linear(mid_features, out_features))

'''
Tiled attention based Attention
'''
class Fusion(nn.Module):
    def __init__(self,v,q,mid,glimpses,dropout = None):
        super(Fusion,self).__init__()
        self.img_conv = nn.Conv2d(v,mid,1,bias=False)
        self.ques_linear = nn.Linear(q,mid)
        self.relu=nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(mid,glimpses,1)
        self.dropout = dropout
        if self.dropout is not None:
            self.drop = nn.Dropout(self.dropout)
        
    def forward(self,v,q):
        if self.dropout is not None:
            v = self.drop(v)
            q = self.drop(q)
        v = self.img_conv(v)#conv layer for image
        q = self.ques_linear(q)#linear layer for text 
        q = self.tile(q,v) # to make visual map and vector of the same dimension
        x = self.relu(v+q) #concatenate and then apply relu
        if self.dropout is not None:
            x = self.drop(x)
        x = self.conv1(x)
        return x
    
    def tile(self,vector,maps):
        s = len(maps.size())-2
        n,c = vector.size()
        vector_new = vector.view(n,c,*([1]*s)).expand_as(maps)
        return vector_new

'''
Average Layer
(Used by tiled attention model)
'''
def average_layer(a,fuse):
    n,c = a.size()[:2]
    s = fuse.size(1)
    a = a.view(n,1,c,-1)
    fuse = fuse.view(n,s,-1)
    fuse = F.softmax(fuse,-1)
    fuse = fuse.unsqueeze(2)
    w = fuse*a
    mean = w.sum(-1)
    return mean.view(n,-1)

'''
Baseline attention
Uses simple one2one multiplciation
'''
class BaselineFusion(nn.Module):
    def __init__(self, num_classes = 3001):
        super(BaselineFusion, self).__init__()
        self.conv1 = nn.Conv2d(2048, 1024, 5)
        self.pool1 = nn.MaxPool2d(3)
        self.relu = nn.ReLU()
        self.FC1 = nn.Linear(4096, num_classes)
    
    def forward(self, x, q):
        x = self.conv1(x)
        x = self.relu(self.pool1(x))
        x = torch.flatten(x,2,3)
        q = q.unsqueeze(2).repeat(1,1,4)
        output = x*q
        output = self.FC1(output.flatten(1))
        return output

'''
Attention block used in stacked attention
model
'''
class AttentionBlock(nn.Module):
    def __init__(self, k):
        super(AttentionBlock, self).__init__()
        self.tanh = nn.Tanh()
        self.lin_img_a = nn.Linear(1024,k, bias = False) # d = 1024
        self.lin_q_a = nn.Linear(1024,k)
        self.lin_p_a = nn.Linear(k,1)
        self.softmax = nn.Softmax(dim = 1)
        
    def forward(self, v_i, v_q):
        # Compute image feature based on cross attention
        # between images and text
        h_q = self.lin_q_a(v_q)
        h_i = self.lin_img_a(v_i)                
        h_a = self.tanh(h_i + h_q)
        
        p_a = self.lin_p_a(h_a)
        p_a = self.softmax(p_a)
        output = v_i*p_a
        output = torch.sum(output, 1)
        return output 

'''
Stacked attention modele
'''
class StackedAttention(nn.Module):
    def __init__(self, num_classes = 3001, k = 120):
        super(StackedAttention, self).__init__()
        self.conv1 = nn.Conv2d(2048, 1024, 3, 1, 1)
        self.relu  = nn.ReLU()

        self.att_block1 = AttentionBlock(k)
        self.att_block2 = AttentionBlock(k)
        
        self.lin_op = nn.Linear(1024, num_classes)
    
    # NOT SURE ABOUT k
    def forward(self, img_f, v_q):
        
        # Convert the Bx2048x10x10 -> Bx1024x10x10
        # note in paper use 512x14x14 -> 512x196 -> 1024x196
        img_f = self.relu(self.conv1(img_f))
        v_i = img_f.flatten(2)
        v_q = v_q.unsqueeze(2)
        
        # Apply 2 levels of stacked attention
        v_1 = self.att_block1(v_i.transpose(1,2), v_q.transpose(1,2))
        v_2 = self.att_block2(v_1, (v_1.unsqueeze(2)+v_q).transpose(1,2))
        
        # Final query to make prediction
        u = v_2+v_q.squeeze(2)
        prediction = self.lin_op(u)
        return prediction