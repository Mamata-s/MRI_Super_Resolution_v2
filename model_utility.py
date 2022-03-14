import torch
import  cv2, os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
# % matplotlib inline




def create_dictionary(image_dir,label_dir):
    lst = []
    for f in os.listdir(image_dir):
        if not f.startswith('.'):
            lst.append(f)
        else:
            pass
    lst.sort()
    label_lst=[]
    for f in os.listdir(label_dir):
        if not f.startswith('.'):
            label_lst.append(f)
        else:
            pass
    label_lst.sort()
    dir_dictionary={}
    for i in range(len(lst)):
        dir_dictionary[lst[i]]=label_lst[i]
        
    return dir_dictionary


def imshow(img,title):
    plt.title(str(img.shape)+title)
    plt.imshow(img,cmap='gray',vmax=1,vmin=0)  # convert from Tensor image. #it takes data in shape (m,n),(m,n,3) or (m,n,4)
    # image having 3 channels is RGB images and images having 4 channels is RGB-A
    # we can convert RGB-A to RGB using image_name.convert('RGB')

def min_max_normalize(image):
    max_img = image.max()
    min_img = image.min()
    denom = max_img-min_img
    norm_image = (image-min_img)/denom
    return norm_image    
    
def save_img_using_pil_lib(img,name,fol_dir):
    data = img
    data = data.astype('float')
    data = (data/data.max())*255
    data = data.astype(np.uint8)
    data = Image.fromarray(data)
    data.save(fol_dir+name+'.png')
    
def save_img(img,name,fol_dir):
    figure(figsize=(8, 6), dpi=80)
    plt.imshow(img, cmap = 'gray')
    plt.axis('off')
    plt.savefig(fol_dir+name+'.png', bbox_inches = 'tight',facecolor='white',pad_inches = 0) 
#     plt.show()    

def batch_min_max_normalize(out_batch):
  x,y,z,p = out_batch.shape
  for i in range(x):
    out_ss = out_batch[i,:,:,:]
    out_ss = out_ss.squeeze().detach().to('cpu').numpy()
    out_ss = min_max_normalize(out_ss)
    out_ss = torch.from_numpy(out_ss)
    out_batch[i,:,:,:] = out_ss
  return out_batch