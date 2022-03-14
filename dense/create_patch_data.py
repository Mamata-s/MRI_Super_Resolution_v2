import os
import numpy as np
from PIL import Image
import torch
import argparse
import utils as ut


def create_index(num_patch, patch_size, xmax, ymax):
    index_list = []
    x_index = torch.randint(0,xmax-patch_size, (num_patch,1))
    y_index = torch.randint(0,ymax-patch_size, (num_patch,1))
    for (x,y) in zip (x_index, y_index):
        index_list.append((x.item(),y.item()))
    return index_list


def create_patch(image_dir, label_dir,dir_dict,patch_size,num_patch,output_image_dir,output_label_dir):
    images = os.listdir(image_dir)

    for index in range(len(images)):
        dict_key = images[index]
        img_path = os.path.join(image_dir,images[index])
        label_path = os.path.join(label_dir, dir_dict[dict_key])
        image = np.array(Image.open(img_path).convert('L'))  #to convert to grayscale
        xmax,ymax = image.shape
        indexes = create_index(num_patch=num_patch,patch_size=patch_size,xmax=xmax,ymax=ymax)
        # image = torch.from_numpy(image)
        label = np.array(Image.open(label_path).convert('L'))
        # label = torch.from_numpy(label)
        for i,idx in enumerate(indexes):
            crop_image,crop_label = extract_patch(image,label,idx,patch_size)
            img_name = dict_key.split(".",1)[0]
            img_name = img_name+'_p'+str(i)
            lbl_name = dir_dict[dict_key].split(".",1)[0]
            lbl_name = lbl_name+'_p'+str(i)
            save_img_using_pil_lib(crop_image,img_name,output_image_dir)
            save_img_using_pil_lib(crop_label,lbl_name,output_label_dir)
    return 1

def extract_patch(img_arr,label_arr,index,patch_size):
    x_index,y_index = index
    image = img_arr [x_index:x_index+patch_size,y_index:y_index+patch_size]
    label = label_arr [x_index:x_index+patch_size,y_index:y_index+patch_size]
    return image,label

def save_img_using_pil_lib(img,name,fol_dir):
    data= img
    data = data.astype('float')
    data = (data/data.max())*255
    data = data.astype(np.uint8)
    data = Image.fromarray(data)
    data.save(fol_dir+name+'.png')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, required=True)
    parser.add_argument('--label-dir', type=str, required=True)
    parser.add_argument('--patch-size', type=int, default=64)
    parser.add_argument('--num-patch', type=int, default=10)
    parser.add_argument('--output-image-dir', type=str, required=True)
    parser.add_argument('--output-label-dir', type=str, required=True)
    args = parser.parse_args()

    dir_dict = ut.create_dictionary(args.image_dir,args.label_dir)
    create_patch (args.image_dir,args.label_dir,dir_dict,args.patch_size,args.num_patch,args.output_image_dir,args.output_label_dir)