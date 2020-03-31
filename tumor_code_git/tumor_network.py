import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
from unet_parts import *

class Flatten(nn.Module):
    def forward(self,input):
        return input.view(input.size(0),-1)

class tumor_densenet121(nn.Module):
    def __init__(self,pretrain):
        super(tumor_densenet121, self).__init__()

        densenet121=models.densenet121(pretrained=pretrain)
        feature_densenet121=densenet121.features
        self.base_model=feature_densenet121
        self.fc1=nn.Linear(in_features=1024*4*4,out_features=1024)
        self.classes=nn.Linear(in_features=1024,out_features=3)
        self.flatten=Flatten()
        self.classifier=nn.Linear(in_features=1024,out_features=3)
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        feature_densenet121=self.base_model(x)

        x=self.flatten(feature_densenet121)
        x = self.fc1(x)
        x=self.relu(x)
        output=self.sigmoid(self.classifier(x))

        return output

class tumor_inceptionv3(nn.Module):
    def __init__(self, pretrain):
        super(tumor_inceptionv3, self).__init__()

        inception_v3 = models.inception_v3(pretrained=pretrain)
        feature_inception_v3 = inception_v3.features
        self.base_model = feature_inception_v3
        self.fc1 = nn.Linear(in_features=1024 * 4 * 4, out_features=1024)
        self.classes = nn.Linear(in_features=1024, out_features=3)
        self.flatten = Flatten()
        self.classifier = nn.Linear(in_features=1024, out_features=3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feature_inception_v3 = self.base_model(x)

        x = self.flatten(feature_inception_v3)
        x = self.fc1(x)
        x = self.relu(x)
        output = self.sigmoid(self.classifier(x))

        return output

class tumor_encoder_cov(nn.Module):
    def __init__(self):
        super(tumor_encoder_cov,self).__init__()

        self.inc = inconv(3, 32)
        self.down1 = down(32,64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 64)
        self.down4 = down(64, 32)

    def forward(self, x):
        en1 = self.inc(x)
        en2 = self.down1(en1)
        en3 = self.down2(en2)
        en4 = self.down3(en3)
        encoded = self.down4(en4)

        return encoded

class tumor_decoder_cov(nn.Module):
    def __init__(self):
        super(tumor_decoder_cov,self).__init__()

        self.up1 = up(32, 64)
        self.up2 = up(64, 128)
        self.up3 = up(128, 64)
        self.up4 = up(64, 32)
        self.outc = outconv(32,3)

    def forward(self, encoded):

        de1 = self.up1(encoded)
        de2 = self.up2(de1)
        de3 = self.up3(de2)
        de4 = self.up4(de3)
        de5 = self.outc(de4)
        decoded=F.tanh(de5)

        return decoded

class tumor_MLP(nn.Module):
    def __init__(self):
        super(tumor_MLP,self).__init__()

        self.fc1=nn.Linear(9*9*32,2048)
        self.fc2=nn.Linear(2048,512)
        self.fc3 = nn.Linear(512,64)
        self.fc4 = nn.Linear(64,3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x=x.view(x.shape[0],-1)
        x=F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        return x