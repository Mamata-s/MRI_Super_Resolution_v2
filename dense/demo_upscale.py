import torch
import torch.nn as nn
import  cv2
import utils as ut
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import dataset as dt
import argparse
from densenet import SRDenseNet, SRDenseNetUpscale
from utils import AverageMeter, calc_psnr,create_dictionary
import numpy as np

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='model demo')
parser.add_argument('-tfactor', type=int, metavar='',required=True,help='trained factor')
parser.add_argument('-factor', type=int, metavar='',required=True,help='resolution factor')
parser.add_argument('-checkpoint', type=str, metavar='',required=True,help='checkpoint path')
parser.add_argument('-name', type=str, metavar='',required=True,help='checkpoint path')
parser.add_argument('--bicubic',
                    help='use dataset from bicubic upsampling', action='store_true')

args = parser.parse_args()


# Load the validation dataset

val_batch_size = 1
str_train_2 = 'crop_factor_2/train'
str_val_2 = 'crop_factor_2/val'
train_img_dir = '../resolution_dataset/dataset/'+ str(str_train_2)
val_img_dir = '../resolution_dataset/dataset/'+str(str_val_2)

train_label_dir = '../resolution_dataset/dataset/label/train'
val_label_dir = '../resolution_dataset/dataset/label/val'

dir_traindict = create_dictionary(train_img_dir,train_label_dir)
val_dir_traindict = create_dictionary(val_img_dir,val_label_dir)

val_label_dir = '../resolution_dataset/dataset/label/val'
dir_valdict = ut.create_dictionary(val_img_dir,val_label_dir)
val_datasets = dt.MRIDatasetUpscale(val_img_dir, val_label_dir, dir_dict = dir_valdict,test=False)
sampler = dt.RdnSampler(val_datasets,val_batch_size,True,classes=val_datasets.classes())
val_dataloaders = torch.utils.data.DataLoader(val_datasets,batch_size = val_batch_size,sampler = sampler,
                    shuffle=False,num_workers=1,pin_memory=False,drop_last=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device available is ',device)

dataiter = iter(val_dataloaders)
images, labels= dataiter.next()

model = SRDenseNetUpscale(num_channels=1, growth_rate=4, num_blocks = 4, num_layers=3).to(device)
print('eval srdense upscale network')


model = nn.DataParallel(model,device_ids=[0,1,2,3])
model.to(device)


state_dict = torch.load(args.checkpoint)
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if 'module' not in k:
        k = 'module.'+k
    else:
        k = k.replace('features.module.', 'module.features.')
    new_state_dict[k]=v

model.load_state_dict(new_state_dict)

model.eval()

images = images.to(device)
labels = labels.to(device)

outputs = model(images)
out = outputs
out = out.squeeze().detach().to('cpu').numpy()
out = ut.min_max_normalize(out)
images = images.squeeze().to('cpu').numpy()
labels = labels.squeeze().detach().to('cpu').numpy()

# plot images
fnsize = 27
fig = plt.figure(figsize=(20,13))

fig.add_subplot(1, 3, 1)
plt.title('Input',fontsize=fnsize)
plt.imshow(images,cmap='gray')
# psnr = peak_signal_noise_ratio(labels,images)
# ssim = structural_similarity( labels,images, multichannel=False)
# plt.xlabel('PSNR=%.2f\nSSIM=%.4f' % (psnr, ssim),fontsize=25)

fig.add_subplot(1, 3, 2)
plt.title('label',fontsize=fnsize)
plt.imshow(labels,cmap='gray')
psnr = peak_signal_noise_ratio(labels, labels)
ssim = structural_similarity(labels, labels, multichannel=False)
plt.xlabel('PSNR=%.2f\nSSIM=%.4f' % (psnr, ssim),fontsize=25)

fig.add_subplot(1, 3, 3)
plt.title('output',fontsize=fnsize)
plt.imshow(out,cmap='gray')
psnr = peak_signal_noise_ratio(labels, out)
ssim = structural_similarity(labels, out, multichannel=False)
plt.xlabel('PSNR=%.2f\nSSIM=%.4f' % (psnr, ssim),fontsize=fnsize)

# fig.add_subplot(1, 5, 4)
# plt.title('Error: label-input')
# error = np.abs(labels-images)
# # error = (labels-images)
# plt.imshow(error,cmap='gray')
# # plt.imshow(error*100,cmap='gray', vmin=0, vmax=1)


# fig.add_subplot(1, 5, 5)
# plt.title('Error : label-output')
# error = np.abs(labels-out)
# # error = (labels-out)
# plt.imshow(error,cmap='gray')
# # plt.imshow(error*100,cmap='gray', vmin=0, vmax=1)

plt.tight_layout()
plt.show()

if args.bicubic:
    save_name =  str(args.name)+'_t'+ str(args.tfactor)+'_v'+str(args.factor)+'_bicubic.png'
else:
    save_name =str(args.name)+'_t'+ str(args.tfactor)+'_v'+str(args.factor)+'.png'
plt.savefig('samples/'+save_name)
    
   

    
    