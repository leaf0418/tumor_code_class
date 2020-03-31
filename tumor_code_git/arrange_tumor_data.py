from glob import glob
import os
import shutil
import cv2
from natsort import natsorted

download_folder='raw_data/download_0324'
download_tumor_path='../tumor_data/'+download_folder
arrange_tumor_path='/home/leaf/leaf_tumor/tumor_data/arrange_tumor'

folder_paths=sorted(glob(download_tumor_path+'/*'))

for folder_path in folder_paths:
    arrange_org_folder_path = arrange_tumor_path + '/train_org_tumor'
    if not os.path.isdir(arrange_org_folder_path):
        os.mkdir(arrange_org_folder_path)
    arrange_mask_folder_path = arrange_tumor_path + '/train_mask_tumor'
    if not os.path.isdir(arrange_mask_folder_path):
        os.mkdir(arrange_mask_folder_path)
    arrange_visual_folder_path = arrange_tumor_path + '/train_visual_tumor'
    if not os.path.isdir(arrange_visual_folder_path):
        os.mkdir(arrange_visual_folder_path)

    folder_name=folder_path.split('/')[-1]
    org_tumor_paths=natsorted(glob(folder_path+'/*.{}'.format('png')))
    mask_tumor_paths =natsorted(glob(folder_path + '/PixelLabelData/*.{}'.format('png')))
    if len(org_tumor_paths)==len(mask_tumor_paths):
        print('process folder:{}'.format(folder_name))

        for i,(org_tumor_path,mask_tumor_path) in enumerate(zip(org_tumor_paths,mask_tumor_paths)):
            number_org_tumor=org_tumor_path.split('/')[-2]
            #copy org image
            dst_org_path=arrange_org_folder_path+'/'+number_org_tumor+'_{:03d}.png'.format(i+1)
            shutil.copy(org_tumor_path, dst_org_path)
            #copy mask image
            dst_mask_path = arrange_mask_folder_path + '/' + number_org_tumor + '_mask_{:03d}.png'.format(i+1)
            shutil.copy(mask_tumor_path, dst_mask_path)
            #copy visual image
            org=cv2.imread(org_tumor_path)
            img = cv2.imread(mask_tumor_path)
            mask = (cv2.imread(mask_tumor_path))[:,:,0]
            org[:,:,1][mask == 1] = 255  # be
            org[:,:,0][mask == 2] = 255  # ma
            org[:, :,2][mask ==3] = 255  # ar
            dst_visual_path = arrange_visual_folder_path + '/' + number_org_tumor + '_visual_{:03d}.png'.format(i + 1)
            cv2.imwrite(dst_visual_path, org)

    else:
        print('{} is error.'.format(folder_name))





print(0)

