from glob import glob
import os
import shutil
import cv2
from natsort import natsorted
dic={'be':0,'ma':1,'ar':2}
path='../tumor_data/classification_test_dataset/'
txt_path=path+'label.txt'

f=open(path+'test_label.txt','a')

folder_paths=sorted(glob(path+'/*'))
for folder_path in folder_paths:
    label=folder_path.split('_')[-1]
    class_paths = sorted(glob(folder_path + '/*'))
    for class_path in class_paths:
        f.write(class_path+' {}\n'.format(dic[label]))
        print(class_path)
f.close()