import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from models.resnet import ResNet50,ResNet34
import os,sys,random
import time
import torch.backends.cudnn as cudnn
from pytorchtools import EarlyStopping
import numpy as np
from tensorboardX import SummaryWriter
from tumor_DataLoader import classification_Datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Trains ResNet on CIFAR or ImageNet',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--dataset',default='tumor')
parser.add_argument('--txt_path',default='../tumor_data/classification_train_dataset/train_label.txt')
parser.add_argument('--batch_size',type=int,default=16)
parser.add_argument('--epochs',type=int,default=500)
parser.add_argument('--workers',type=int,default=0)
parser.add_argument('--log',default='../tumor_data/classification_train_dataset')
parser.add_argument('--teacher_backbone',default='Resnet50')
parser.add_argument('--seed',type=int,default=2020)
parser.add_argument('--num_classes',type=int,default=3)
parser.add_argument('--input_size',type=int,default=32)
args = parser.parse_args()

log_time='{:02d}{:02d}{:02d}{:02d}'.format(time.localtime()[1],time.localtime()[2],time.localtime()[3],time.localtime()[4])
log_folder=args.log+'/log_'+log_time
if not os.path.isdir(log_folder):
    os.mkdir(log_folder)

log_txt = '{}'.format(os.path.join(log_folder, 'log_'+log_time+'.txt'))
log=open(log_txt,'w')
tensorX = '{}'.format(os.path.join(log_folder, 'tensorboard_'+log_time))
writer=SummaryWriter(log_dir=tensorX)



device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
best_loss=np.inf
best_acc=0.0

def get_random_seed(num=2020):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic=True

def tumor_dataset():
    #dataset
    train_dataset=classification_Datasets(args.txt_path,args.input_size)
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    shuffle_dataset = True

    if shuffle_dataset:
        np.random.seed(2020)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=1, sampler=train_sampler)
    validation_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=1, sampler=validation_sampler)
    return train_loader,validation_loader

#Model
if args.dataset=='cifar10':
    numberclass=10
if args.dataset=='cifar100':
    numberclass=100

net=ResNet50(classnum=args.num_classes)


rand_input=torch.rand(args.batch_size,3,64,64)
writer.add_graph(net,(rand_input,))

net = net.to(device)


if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)

def getdata():
    if args.dataset=='cifar10':
        mean=[x/255 for x in [125.3,123.0,113.9]]
        std=[x/255 for x in [63.0,62.1,66.7]]
        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5), transforms.ToTensor()
                , transforms.Normalize(mean, std)])
        transform_val = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=False, transform=transform_train)
        valset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_val)
        numclass=10


    if args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5), transforms.ToTensor()
                , transforms.Normalize(mean, std)])
        transform_val = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        trainset = torchvision.datasets.CIFAR100(root='./', train=True, download=True, transform=transform_train)
        valset = torchvision.datasets.CIFAR100(root='./', train=False, download=True, transform=transform_val)
        numclass = 100


    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False,num_workers=args.workers, pin_memory=True)


    return train_loader,val_loader


#Training
def train(epoch):
    global log
    train_loader,val_loader=tumor_dataset()
    print_log('\nEpoch:{:d}'.format(epoch), log)
    net.train()
    train_loss=0
    correct=0
    total=0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # writer.add_image('image', targets, inputs, epoch)
        inputs, targets = inputs.to(device),targets.to(device)
        optimizer.zero_grad()
        outputs=net(inputs)
        loss=criterion(outputs,targets)
        loss.backward()
        optimizer.step()

        train_loss+=loss.item()
        _,predicted=outputs.max(1)
        total+=targets.size(0)
        correct+=predicted.eq(targets).sum().item()
    print_log('Teacher Train  |Batch_idx:{:<3d}|Train Loss:{:<8.3f}|Train Acc:{:<8.3f}'.format(batch_idx, (train_loss / (batch_idx + 1)),100. * correct / total),log)
    writer.add_scalar('train/loss',(train_loss / (batch_idx + 1)),epoch)
    writer.add_scalar('train/acc', (100. * correct / total), epoch)

def validation(epoch):
    global  log,best_loss,best_acc
    train_loader, val_loader= tumor_dataset()
    net.eval()
    val_loss=0
    correct=0
    total=0
    early_stop = EarlyStopping(patience=10,verbose=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        if best_acc<100. * correct / total:
            best_acc=100. * correct / total
            print_log('Update best acc : {:<5.3f}'.format(best_acc),log)
        if (val_loss / (batch_idx + 1)) < best_loss:
            best_loss = (val_loss / (batch_idx + 1))
            print_log('Save best model | Loss : {}| Acc : {}'.format(val_loss / (batch_idx + 1), 100. * correct / total), log)
            torch.save(net, './{}/{}_model.pth'.format(log_folder, args.teacher_backbone))
            torch.save(net.state_dict(), './{}/{}_weight.pth'.format(log_folder, args.teacher_backbone))
        print_log('Teacher Val    |Batch_idx:{:<3d}|Val Loss  :{:<8.3f}|Val Acc:{:<8.3f}'.format(batch_idx, (val_loss / (batch_idx + 1)), 100. * correct / total),log)
        writer.add_scalar('val/loss', (val_loss / (batch_idx + 1)), epoch)
        writer.add_scalar('val/acc', (100. * correct / total), epoch)
        early_stop(val_loss, net)
        while early_stop.early_stop:
            print_log("Early stop",log)
            writer.close()
            log.close()
            break

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def main():
    global best_acc,best_loss
    print_log('create log ==> {}'.format(log_txt), log)
    print_log('=' * 56 + '\n BACKBONE : {}\n'.format(args.teacher_backbone) + '=' * 56, log)
    print_log("MODEL SETTING",log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)
    print_log('=' * 56, log)
    print_log("teacher backbone : {}".format(args.teacher_backbone), log)
    print_log("dataset : {}".format(args.dataset), log)
    print_log("batch size : {}".format(args.batch_size), log)
    print_log("epochs : {}".format(args.epochs), log)
    print_log('=' * 56, log)

    get_random_seed(args.seed)

    for epoch in range(start_epoch, start_epoch+args.epochs):
        train(epoch)
        validation(epoch)
        scheduler.step(10)
    print_log('Best model | Loss : {:<8.3f}| Acc : {:<8.3f}'.format(best_loss, best_acc), log)
    writer.close()
    log.close()


if __name__=="__main__":
    main()










