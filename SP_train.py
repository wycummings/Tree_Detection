import argparse
import os
import pdb
import time

import torch
import torch.nn     as nn
import torch.optim  as optim
import pandas       as pd

#from model import SampleNet
from   vgg          import vgg_16
from   Resnet       import resnet18
from   Resnet       import resnet34
from   Resnet       import resnet50
from   Resnet       import incep_v3
from   utils        import make_dataloader


#import tensorboardX as tbx


# Training settings
parser = argparse.ArgumentParser(description='sample script of training model classifying tree patches')
parser.add_argument('exp_name', help='name of this experiment(output folder)')
parser.add_argument('--data_dir',  type=str,   default='./new_dataset', help='directory contains training data.')
parser.add_argument('--nClass',    type=int,   default=7,           help='number of prediction class')
parser.add_argument('--nEpochs',   type=int,   default=100,         help='number of epochs to train for')
parser.add_argument('--patchsize', type=int,   default=224,         help='training patch size')
parser.add_argument('--batchsize', type=int,   default=32,        help='training/test batch size')
parser.add_argument('--lr',        type=float, default=0.001,        help='learning rate. Default=0.003')
parser.add_argument('--wd',        type=float, default=0.0001,       help='L2-weight decay. Default=0.001')
args = parser.parse_args()
print(args)

device = torch.device('cuda')


print('===> Loading datasets')
train_dataloader   = make_dataloader(os.path.join(args.data_dir, 'train'), args.batchsize, args.patchsize)
train_SPdataloader = make_dataloader(os.path.join(args.data_dir, 'train'), args.batchsize, args.patchsize, SP=True)
val_dataloader     = make_dataloader(os.path.join(args.data_dir, 'val'),   args.batchsize, args.patchsize, val=True)


print('===> Building model')
model     = resnet34(args.patchsize, args.nClass).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=args.wd, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True, min_lr=0.00001)
#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,80,145], gamma=0.1)


print('---------- Networks initialized -------------')
print(model)
print('-----------------------------------------------')




def train(epoch, tally):
    model.train()
    avg_loss = 0
    correct  = 0
    total    = 0
    #
    if epoch % 10 == 9 or epoch % 10 == 0 or epoch < 0.1*args.nEpochs or epoch > .9*args.nEpochs:
        dataloader = train_dataloader
    elif tally >= 10:
        print('tallySP')
        dataloader = train_SPdataloader
    else:
        print('SP')
        dataloader = train_SPdataloader

        
    
    for iteration, batch in enumerate(dataloader, 1):
            # forward
        input, target = batch
        input, target = input.to(device), target.to(device)
        prediction    = model(input)
        loss          = criterion(prediction, target)
        avg_loss     += loss.item()
        _, predicted  = torch.max(prediction, 1)
        total        += target.size(0)
        correct      += (predicted == target).sum().item()
        
        
        # Update network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = avg_loss / len(dataloader)
    print("---Epoch[{}]---\n===> Avg. Loss: {:.4f}".format(epoch, avg_loss))
    print('===> Training Accuracy: %d %%\n' % (100 * correct / total))
    return avg_loss


def validate():
    start    = time.time()
    model.eval()
    avg_loss = 0
    correct  = 0
    total    = 0
    for batch in val_dataloader:
        with torch.no_grad():
            input, target = batch
            input, target = input.to(device), target.to(device)
            prediction    = model(input)
            avg_loss     += criterion(prediction, target).item()
            _, predicted  = torch.max(prediction, 1)
            total        += target.size(0)
            correct      += (predicted == target).sum().item()
    
    avg_loss = avg_loss / len(val_dataloader)
    print('=====> Test Avg. Loss: {:.4f}'.format(avg_loss))
    print('=====> Validation Accuracy: %d %%\n' % (
        100 * correct / total))
    print('elapsed time: {}sec'.format(time.time()-start))
    return avg_loss


def save_model(epoch, loss):
    os.makedirs('saved_models', exist_ok=True)

    path   = 'saved_models/{}.pth'.format(args.exp_name)
    torch.save(model, path)
    print('Best model saved to {}\n'.format(path))
    
    with open(os.path.join('saved_models', 'log'), 'a') as f:
        f.write('{}[Epoch:{:>3}] loss: {:.4f}\n'.format(args.exp_name, epoch, loss))

pdb.set_trace()
#write = tbx.SummaryWriter()
min_loss = 100
min_tally = 0
Train_log  = pd.DataFrame(columns=['epoch','training loss', 'val loss'])
for epoch in range(1, args.nEpochs + 1):
    train_loss = train(epoch, tally=min_tally)
    val_loss   = validate()
    if epoch == .8*args.nEpochs:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True, min_lr=0.000001)
    scheduler.step(val_loss)
    log = pd.DataFrame([(epoch,train_loss,val_loss)], columns=['epoch', 'training loss', 'val loss'])
    Train_log = pd.concat([Train_log,log])
#    write.add_scalars('data/total_loss', {
                       # "train":train_loss,
                        #"val":val_loss}
                        #, epoch)
    
    if val_loss < min_loss:
        save_model(epoch, val_loss)
        min_loss = val_loss
        min_tally = 0
    else:
        min_tally += 1
#write.export_scalars_to_json("./all_scalars.json")
#write.close()   
Train_log.to_csv('TrainVal_data{}.csv'.format(args.exp_name[2:]), index=False)

