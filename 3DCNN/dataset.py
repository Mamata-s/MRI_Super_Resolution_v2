import torch
import  cv2, os
from torch.utils.data import Dataset
import numpy as np

import numpy as np
import nibabel as nib
import torch
import warnings
warnings.filterwarnings("ignore")

def min_max_normalize(image):
    max_img = image.max()
    min_img = image.min()
    # denom = (max_img-min_img) + 0.00000000001
    # norm_image = (image-min_img)/denom
    denom = max_img + 0.00000000001
    norm_image = image/denom
    return norm_image 

def create_index(db_size, cube_size, xmax, ymax, zmax):
    index_list = []
    x_index = torch.randint(0,xmax-cube_size, (db_size,1))
    y_index = torch.randint(0,ymax-cube_size, (db_size,1))
    z_index = torch.randint(0,zmax-cube_size, (db_size,1))
    for (x,y,z) in zip (x_index, y_index, z_index):
        index_list.append((x.item(),y.item(),z.item()))
    return index_list

def load_data_nii(fname):
    img = nib.load(fname)
    # affine_mat=img.affine
    # hdr = img.header
    data = img.get_fdata()
    data_norm = torch.from_numpy(data)
    return data_norm 


def preprocess_data(data,factor=2,pad=True):
    spectrum_3d = np.fft.fftn(data)  # Fourier transform along Y, X and T axes to obtain ky, kx, f
    spectrum_3d_sh = np.fft.fftshift(spectrum_3d, axes=(0,1,2))  # Apply frequency shift along spatial dimentions so

    x,y,z = spectrum_3d_sh.shape
    data_pad = np.zeros((x,y,z),dtype=np.complex_)
    center_y = y//2 #defining the center of image in x and y direction
    center_x = x//2
    center_z = z//2
    startx = center_x-(x//(factor*2))  
    starty = center_y-(y//(factor*2))
    startz = center_z-(z//(factor*2))
    arr = spectrum_3d_sh[startx:startx+(x//factor),starty:starty+(y//factor),startz:startz+(z//factor)]
    # arr = spectrum_3d_sh[startx:startx+(x//factor),starty:starty+(y//factor),:]
    # arr = spectrum_3d_sh    
    if pad:
        data_pad[startx:startx+(x//factor),starty:starty+(y//factor),startz:startz+(z//factor)] = arr
        img_reco_cropped = np.fft.ifftn(np.fft.ifftshift(data_pad))
    else:
        img_reco_cropped = np.fft.ifftn(np.fft.ifftshift(arr)) 
    return np.abs(img_reco_cropped)


class MRI3DDataset(Dataset):
    def __init__(self,image_arr,label_arr,indexes=None,cube_size=None, eval=False):
        self.indexes = indexes
        self.label_arr = label_arr
        self.image_arr = image_arr
        self.cube_size = cube_size
        self.eval = eval
        
    def __len__(self):
        if self.indexes:
            return len(self.indexes)
        else:
            return 1
    
    def __getitem__(self, index):
        if self.eval:
            image = self.image_arr 
            label = self.label_arr 
        else:
            x_index,y_index,z_index = self.indexes[index]
            image = self.image_arr [x_index:x_index+self.cube_size,y_index:y_index+self.cube_size,z_index:z_index+self.cube_size]
            label = self.label_arr [x_index:x_index+self.cube_size,y_index:y_index+self.cube_size,z_index:z_index+self.cube_size]
       
        image = min_max_normalize(image)
        label = min_max_normalize(label)
        
        image = torch.from_numpy(image)
        # label = torch.from_numpy(label)

        image= torch.unsqueeze(image,0)
        label = torch.unsqueeze(label,0)

        return image,label