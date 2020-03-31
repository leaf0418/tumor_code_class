from glob import glob
import os
import shutil
import cv2
from natsort import natsorted
import numpy as np
from skimage import measure

path='/home/leaf/leaf_tumor/tumor_data/arrange_tumor'
org_path='/home/leaf/leaf_tumor/tumor_data/arrange_tumor/test_org_tumor'
mask_path='/home/leaf/leaf_tumor/tumor_data/arrange_tumor/test_mask_tumor'
visual_path='/home/leaf/leaf_tumor/tumor_data/arrange_tumor/test_visual_tumor'

sort_org_paths=natsorted(glob(org_path+'/*'))
sort_mask_paths=natsorted(glob(mask_path+'/*'))
sort_visual_paths=natsorted(glob(visual_path+'/*'))

# kernel=np.ones((4,4),np.uint8)
dict={'be':1,'ma':2,'ar':3}
num_be,num_ma,num_ar=0,0,0

if len(sort_org_paths)==len(sort_mask_paths):
    for num_img,(org_img_path,mask_img_path,visual_path) in enumerate(zip(sort_org_paths,sort_mask_paths,sort_visual_paths)):
        #read img
        raw_org_img=cv2.imread(org_img_path)
        raw_mask_img = cv2.imread(mask_img_path)
        raw_visual_img = cv2.imread(visual_path)
        to_visual_img=raw_org_img.copy()

        mask_img=raw_mask_img[:,:,0]
        to_visual_img[:, :, 1][mask_img == dict['be']] = 255  # be
        to_visual_img[:, :, 0][mask_img == dict['ma']] = 255  # ma
        to_visual_img[:, :, 2][mask_img == dict['ar']] = 255  # ar

        for tumor_classes in range(1, 4):
            tmp_black = np.zeros((raw_org_img.shape[0],raw_org_img.shape[1]),dtype=int)
            tmp_black[mask_img == tumor_classes] = tumor_classes
            labels = measure.label(tmp_black, connectivity=1, return_num=True)

            for num_tumor in range(1, 8):
                map = np.zeros((raw_org_img.shape[0],raw_org_img.shape[1]),dtype=int)
                map[labels[0] == num_tumor] = tumor_classes
                sum_value = np.sum(map == tumor_classes)
                if sum_value>=100:
                    print('sum: {:>6d} | tumor classes: {:>3d} | number: {:>3d}'.format(sum_value,tumor_classes, num_tumor))
                    non_zero=np.where(map!=0)
                    crop_raw_img=raw_org_img[np.min(non_zero[0])-10:np.max(non_zero[0])+10,np.min(non_zero[1])-10:np.max(non_zero[1])+10,:]
                    crop_visual_img=to_visual_img[np.min(non_zero[0])-10:np.max(non_zero[0])+10,np.min(non_zero[1])-10:np.max(non_zero[1])+10,:]

                    if tumor_classes==dict['be']:
                        num_be=num_be+1
                        split_org_be=path+'/spilt_test_org_be'
                        if not os.path.isdir(split_org_be):
                            os.mkdir(split_org_be)
                        cv2.imwrite(split_org_be+'/be_{:04d}.png'.format(num_be),crop_raw_img)
                        split_visual_be = path + '/spilt_test_visual_be'
                        if not os.path.isdir(split_visual_be):
                            os.mkdir(split_visual_be)
                        cv2.imwrite(split_visual_be +'/be_{:04d}.png'.format(num_be),crop_visual_img)

                    if tumor_classes == dict['ma']:
                        num_ma = num_ma + 1
                        split_org_ma = path + '/spilt_test_org_ma'
                        if not os.path.isdir(split_org_ma):
                            os.mkdir(split_org_ma)
                        cv2.imwrite(split_org_ma + '/ma_{:04d}.png'.format(num_ma), crop_raw_img)
                        split_visual_ma = path + '/spilt_test_visual_ma'
                        if not os.path.isdir(split_visual_ma):
                            os.mkdir(split_visual_ma)
                        cv2.imwrite(split_visual_ma + '/ma_{:04d}.png'.format(num_ma), crop_visual_img)

                    if tumor_classes == dict['ar']:
                        num_ar = num_ar + 1
                        split_org_ar = path + '/spilt_test_org_ar'
                        if not os.path.isdir(split_org_ar):
                            os.mkdir(split_org_ar)
                        cv2.imwrite(split_org_ar+ '/ar_{:04d}.png'.format(num_ar), crop_raw_img)
                        split_visual_ar = path + '/spilt_test_visual_ar'
                        if not os.path.isdir(split_visual_ar):
                            os.mkdir(split_visual_ar)
                        cv2.imwrite(split_visual_ar + '/ar_{:04d}.png'.format(num_ar), crop_visual_img)


                        # cv2.imshow('num_img:{}'.format(num_img), to_visual_img)
                        # cv2.imshow('raw |tumor:{}|number:{}'.format(tumor_classes, num_tumor), crop_raw_img)
                        # cv2.imshow('visual |tumor:{}|number:{}'.format(tumor_classes, num_tumor), crop_visual_img)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()

                    # cv2.imshow('num_img:{}'.format(num_img),to_visual_img)
                    # cv2.imshow('raw |tumor:{}|number:{}'.format(tumor_classes, num_tumor),crop_raw_img)
                    # cv2.imshow('visual |tumor:{}|number:{}'.format(tumor_classes, num_tumor),crop_visual_img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()


else:
    print('The length is different.')

