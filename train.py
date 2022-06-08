'''
Just moved the save checkpoint after 
code so that if session gets killed it starts
at epoch before starts so that we don't loose a data-point on eval.

Main train script.

NOTE - Remember to give an unique name so that previous training
results are not overwritten

Other available options -
1. epochs
2. learning_rate
3. batch_size
4. num_classes
5. experiment
6. checkpoint_dir - path to .pt saved checkpoint.
7  model - SAN/Tiled/Baseline

Note - Output saved as following 

exp_dir_name
    -models <contains .pth model>
    -checkpoint.pt
    -train/val loss/acc .npy
    
 To run
 python train.py --experiment <exp_dir_name>
 
 To run from saved checkpoint
 python train.py --checkpoint_dir experiment/checkpoint.pt
 
'''

import torch
from Net import TiledAttention, BaseLine , SANModel
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
from VQA_Dataset import VQA_Data
import json
import torch.optim as optim
import os
import os.path as osp
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import argparse

#------------------------------------------------------------------------------------------
def run_one_epoch(model, dataloader, iter, epoch, toTrain = True):
    loss_arr = []
    acc_arr = []
    if toTrain:
        op_name = 'Train'
        model.train()
    else:
        op_name = 'Val'
        model.eval()
        
    for i, d_batch in enumerate(dataloader):
        iter += 1
        
        # Prepare data
        ques_e = d_batch['quest_e'].squeeze(1).to(device)
        img_e = d_batch['img_e'].squeeze(1).to(device)
        ques_l = d_batch['quest_l']
        ans_e = d_batch['ans_e'].view(-1).to(device)
        
        if toTrain:
            # train
            optimizer.zero_grad()
        
        # forward pass
        pred = model(ques_e, img_e, ques_l)
        loss = ce(pred, ans_e)
        
        if toTrain:
            # train
            loss.backward()        
            optimizer.step()
            
        # accuracy
        acc = torch.sum(torch.argmax(pred, 1) == ans_e).cpu().data.item()/batch_size
        loss_arr.append(loss.item())
        acc_arr.append(acc)
        
        # Print progress
        print(op_name + ' Epoch %d Iter %d Loss : %s' %(epoch+1, iter, loss.cpu().data.item()))
        print(op_name + ' Epoch %d Iter %d Acc : %s' %(epoch+1, iter, acc))
        
    return np.mean(np.array(loss_arr)), np.mean(np.array(acc_arr))
#------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
# The location of training set
parser.add_argument('--experiment', default='experiment', help='path to save checkpoint' )
parser.add_argument('--epochs', type=int, default=50, help='the number of epochs being trained')
parser.add_argument('--learning_rate', type=float, default=0.01, help='the initial learning rate')
parser.add_argument('--batchSize', type=int, default=16, help='the size of a batch' )
parser.add_argument('--num_classes', type=int, default=3001, help='the number of classes' )
parser.add_argument('--checkpoint_dir', default=None, help='path to checkpoints' )
parser.add_argument('--optim', default='adam', help='select optimizer' )
parser.add_argument('--dropout', type=float, default=None, help='dropout' )
parser.add_argument('--model', type=str, default='Tiled', help='Select model type (SAN/Tiled/Baseline)')
parser.add_argument('--vocabSize', type=int, default=1000, help='specify size of vocab 1000/3000' )

# The detail network setting
opt = parser.parse_args()

# Save hyperparameters
hparams = {}
hparams['batchSize'] = opt.batchSize
hparams['learningRate'] = opt.learning_rate
hparams['optim'] = opt.optim
hparams['dropout'] = opt.dropout
hparams['epochs'] = opt.epochs
hparams['model'] = opt.model
hparams['vocabSize'] = opt.vocabSize

print('*****************')
print(hparams)

# Training settings
END_EPOCH = opt.epochs
START_EPOCH = 0
learning_rate = opt.learning_rate
batch_size = opt.batchSize
num_classes = opt.num_classes
dropout = opt.dropout

# Handle paths to save model, checkpoints, loss & acc
exp_dir = osp.join('trained_model', opt.experiment)
chekpoint_path = osp.join(exp_dir, 'checkpoint.pt')
model_path = osp.join(exp_dir, 'models')

if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
    os.mkdir(model_path)
    hparmas_file = osp.join(exp_dir, 'hyper_params.json')
    with open(hparmas_file, 'w') as fp:
        json.dump(hparams, fp)
    

# load vocab
if opt.vocabSize == 3000:
    l_dict = json.load(open('data/vocabulary_alternate.json'))
else:
    l_dict = json.load(open('data/vocabulary_1000.json'))

# Create train dataloader
dataset = VQA_Data('train', opt.vocabSize)
train_dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True)

# Create val dataloader
dataset = VQA_Data('val', opt.vocabSize)
val_dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True)

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model
if opt.model == "Tiled":
    model = TiledAttention(len(l_dict['questions'])+1, dropout).to(device)
elif opt.model == "SAN":
    model = SANModel(len(l_dict['questions'])+1).to(device)
elif opt.model == "Baseline":
    model = BaseLine(len(l_dict['questions'])+1).to(device)
else:
    print("Model type from SAN/Tiled/Baseline")
    
# Optimizer
if opt.optim == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
else:
    optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)

# exponential learning rate
decayRate = 0.9
my_lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

# Loss
ce = nn.CrossEntropyLoss()

# Main training code
iter = 0
TRAIN_ACC = []
TRAIN_LOSS = []
VAL_LOSS = []
VAL_ACC = []

# Load checkpoint if available
if opt.checkpoint_dir is not None:
    print('Loading saved checkpoint ......')
    saved_chkpt = torch.load(opt.checkpoint_dir)
    model.load_state_dict(saved_chkpt['model_state_dict'])
    optimizer.load_state_dict(saved_chkpt['optimizer_state_dict'])
    TRAIN_LOSS = saved_chkpt['train_loss']
    TRAIN_ACC = saved_chkpt['train_acc']
    VAL_LOSS = saved_chkpt['val_loss']
    VAL_ACC = saved_chkpt['val_acc']
    iter = saved_chkpt['iter']
    START_EPOCH = saved_chkpt['epoch'] + 1
    

EPOCH_RANGE = np.arange(START_EPOCH, END_EPOCH)
    
for epoch in tqdm(EPOCH_RANGE):
    # Main training loop
    train_loss, train_acc =  run_one_epoch(model, train_dataloader, iter, epoch, True)
    TRAIN_LOSS.append(train_loss)
    TRAIN_ACC.append(train_acc)
    
    # Save loss and acc
    np.save('%s/train_loss.npy' % (exp_dir), np.array(TRAIN_LOSS))
    np.save('%s/train_acc.npy' % (exp_dir), np.array(TRAIN_ACC))
        
    if (epoch+1) % 3 == 0:
        # Perform validation and save model  checkpoint
        # Validation code
        val_loss, val_acc = run_one_epoch(model, val_dataloader, 0, 0,  False)
        VAL_LOSS.append(val_loss)
        VAL_ACC.append(val_acc)
        # Save val loss and acc
        np.save('%s/val_loss.npy' % (exp_dir), np.array(VAL_LOSS))
        np.save('%s/val_acc.npy' % (exp_dir), np.array(VAL_ACC))
        
        # Save model checkpoint
        torch.save(model.state_dict(), '%s/net_%d.pth' % (model_path, epoch+1) )
        
    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': TRAIN_LOSS,
        'train_acc': TRAIN_ACC,
        'val_loss': VAL_LOSS,
        'val_acc': VAL_ACC,
        'iter': iter
        }, chekpoint_path)