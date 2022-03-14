import numpy as np
import math


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calc_psnr(img1, img2):
    return 10. * np.log10(1. / np.mean((img1 - img2) ** 2))

def calculate_l1_distance(img,img2):
    dist = np.sum(abs(img[:] - img2[:]));
    return dist
 
def calculate_l2_distance(img,img2):
    dist = np.sqrt(np.sum((img[:] - img2[:])** 2));
    return dist
 
def calculate_RMSE(img,img2):
    m=img.shape[0]
    n= img.shape[1]
    rmse = np.sqrt(np.sum((img[:] - img2[:])** 2)/(m*n));
    return rmse
 
def calculate_PSNR(img,img2):
    psnr = 10* math.log10( (np.sum(img[:]** 2)) / (np.sum((img[:] - img2[:])** 2)) )
    return psnr

def Average(lst):
    return sum(lst) / len(lst)