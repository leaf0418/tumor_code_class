
import argparse
import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import numpy as np
from tumor_DataLoader import classification_Datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Trains ResNet on CIFAR or ImageNet',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--dataset',default='tumor')
parser.add_argument('--txt_path',default='../tumor_data/classification_train_dataset/train_label.txt')
parser.add_argument('--batch_size',type=int,default=32)
parser.add_argument('--epochs',type=int,default=500)
parser.add_argument('--workers',type=int,default=0)
parser.add_argument('--log',default='../tumor_data/classification_train_dataset')
parser.add_argument('--teacher_backbone',default='Resnet50')
parser.add_argument('--seed',type=int,default=2020)
parser.add_argument('--num_classes',type=int,default=3)
parser.add_argument('--pretrained',type=bool,default=True)
parser.add_argument('--input_size',type=int,default=64)
args = parser.parse_args()

##setting inital
device = 'cuda' if torch.cuda.is_available() else 'cpu'




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


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


T=transforms.Compose(
    [transforms.Resize((args.input_size, args.input_size)), transforms.RandomHorizontalFlip(0.5), transforms.ToTensor()
        , transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

gt_label=[]
pre_label=[]


def test():
    # Model
    net = torch.load('../tumor_data/classification_train_dataset/log_03250233/Resnet50_model.pth')
    net=net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    torch.no_grad()
    with open('/home/leaf/leaf_tumor/tumor_data/classification_test_dataset/test_label.txt', 'r') as f:
        for line in f.readlines():
            string = line.split(' ')
            raw_test_img=Image.open('{}'.format(string[0]))
            test_img=T(raw_test_img).unsqueeze(0)
            test_img_cuda=test_img.to(device)
            output=net(test_img_cuda)
            _,output_label=torch.max(output,1)
            pre_label.append(output_label.item())
            gt_label.append(int(string[1]))
        confmat=confusion_matrix(y_true=gt_label,y_pred=pre_label)
        fig,ax=plt.subplots(figsize=(4,4))
        ax.matshow(confmat)
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                ax.text(x=j,y=i,s=confmat[i,j],va='center',ha='center')
        plt.xlabel('pre label')
        plt.ylabel('gt label')
        plt.show()

            # print(output_label.item(),int(string[1]))


# def main():
#     print()



if __name__=="__main__":

    test()











