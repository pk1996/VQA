'''
Generates ResNet-152 image features. Need to specify 
the image and output dir.

Note - Images are cropped 300x300 at top left.
'''
import argparse
import torch
import torch.nn as nn
import torchvision
import os
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--imgRoot', default = '../data/train/train2014/', help = 'path to images')
parser.add_argument('--outRoot', default = '../data/train/img_features/', help = 'path to output features')

opt = parser.parse_args()

# Create Resnet-152 pre-trained model
resnet152 = torchvision.models.resnet152(pretrained= True)
# Features extracted at 8th layer
new_module = list(resnet152.children())[:8]
resnet152 = torch.nn.Sequential(*new_module).cuda()

if not os.path.exists(opt.outRoot):
    os.mkdir(opt.outRoot)

# Iterate through images
img_list = sorted(os.listdir(opt.imgRoot))
for img_name in tqdm(img_list):
    # Read image
    img_path = osp.join(opt.imgRoot, img_name)
    img = cv2.imread(img_path)
    img = img[:300,:300,:] # Crop out a 300x300 patch
    
    # Convert to tensor
    img_tensor = torch.tensor(np.expand_dims(img, 0)).cuda()
    img_tensor = img_tensor.permute([0,3,1,2])
    img_tensor = img_tensor.to(torch.float)
    
    # Extract features
    img_feature = resnet152(img_tensor)
    
    # Save tensor to file
    tensor_f_name = img_name
    tensor_f_name = tensor_f_name.split('.')[0] + '.pt'
    torch.save(img_feature, osp.join(opt.outRoot, tensor_f_name))