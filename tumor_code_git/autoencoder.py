import os,sys,random,time
import argparse
import torch
import torch.nn as nn
from tumor_DataLoader import classification_Datasets
import torch.backends.cudnn as cudnn
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import numpy as np
from torch.utils.data import DataLoader
from tumor_network import tumor_decoder_cov,tumor_encoder_cov
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Trains ResNet on CIFAR or ImageNet',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--dataset',default='tumor')
parser.add_argument('--txt_path',default='../tumor_data/classification_train_dataset/train_label.txt')
parser.add_argument('--batch_size',type=int,default=128)
parser.add_argument('--epochs',type=int,default=500)
parser.add_argument('--workers',type=int,default=0)
parser.add_argument('--log',default='../tumor_data/classification_train_dataset')
parser.add_argument('--backbone',default='autoencoder')
parser.add_argument('--seed',type=int,default=2020)
parser.add_argument('--num_classes',type=int,default=3)
parser.add_argument('--pretrained',type=bool,default=True)
parser.add_argument('--input_size',default=144)
args = parser.parse_args()

##setting folder
log_time='{:02d}{:02d}{:02d}{:02d}'.format(time.localtime()[1],time.localtime()[2],time.localtime()[3],time.localtime()[4])
log_folder=args.log+'/log_'+log_time+'auto'
if not os.path.isdir(log_folder):
    os.mkdir(log_folder)
log_txt = '{}'.format(os.path.join(log_folder, 'log_'+log_time+'.txt'))
log=open(log_txt,'w')
tensorX = '{}'.format(os.path.join(log_folder, 'tensorboard_'+log_time))
writer=SummaryWriter(log_dir=tensorX)

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

print_log('create log ==> {}'.format(log_txt), log)
print_log('=' * 56 + '\n BACKBONE : {}\n'.format(args.backbone) + '=' * 56, log)
print_log("MODEL SETTING",log)
print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
print_log("torch  version : {}".format(torch.__version__), log)
print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)
print_log('=' * 56, log)
print_log("teacher backbone : {}".format(args.backbone), log)
print_log("dataset : {}".format(args.dataset), log)
print_log("batch size : {}".format(args.batch_size), log)
print_log("epochs : {}".format(args.epochs), log)
print_log('=' * 56, log)


N_TEST_IMG=5
best_loss=np.inf

#dataset
train_dataset=classification_Datasets(args.txt_path,args.input_size)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=1, shuffle=True)
print(train_dataset.transform_img)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#net
encoder=tumor_encoder_cov()
decoder=tumor_decoder_cov()


encoder = encoder.to(device)
if device == 'cuda':
    encoder = torch.nn.DataParallel(encoder)
    decoder = torch.nn.DataParallel(decoder)
    cudnn.benchmark = True
para=list(encoder.parameters())+list(decoder.parameters())
optimizer=torch.optim.Adam(para,lr=args.lr)
loss_func=nn.MSELoss()

for epoch in range(args.epochs):
    for step,(inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        encoded=encoder(inputs)
        decoded = decoder(encoded)

        loss=loss_func(decoded,inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if best_loss>=loss.data.item():
        print('best loss:{:.4f} | new loss:{:.4f}'.format(best_loss,loss.data.item()))
        best_loss=loss.data.item()
        print_log('Save model | best loss : {:.4f}'.format(loss.data.item()),log)
        torch.save(encoder, './{}/{}_en_model.pth'.format(log_folder, args.backbone))
        torch.save(encoder.state_dict(), './{}/{}_en_weight.pth'.format(log_folder, args.backbone))
        torch.save(decoder, './{}/{}_de_model.pth'.format(log_folder, args.backbone))
        torch.save(decoder.state_dict(), './{}/{}_de_weight.pth'.format(log_folder, args.backbone))
    writer.add_scalar('train/loss', (loss.data.item() / (step + 1)), epoch)
    print_log('epoch:{:03d} | step:{:03} | train loss:{:.4f}'.format(epoch,step, loss.data.item()),log)

writer.close()
log.close()

