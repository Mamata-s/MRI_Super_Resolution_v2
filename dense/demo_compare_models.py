import torch
import torch.nn as nn
import  cv2
import utils as ut
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import dataset as dt
import argparse
from densenet import SRDenseNet
import numpy as np

import warnings
warnings.filterwarnings("ignore")



def load_model(checkpoint):
    model= SRDenseNet(num_channels=1, growth_rate=4, num_blocks = 4, num_layers=3).to(device)

    model = nn.DataParallel(model,device_ids=[0,1,2,3])
    model.to(device)

    state_dict = torch.load(checkpoint)
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
    return model



parser = argparse.ArgumentParser(description='model demo')
parser.add_argument('-factor', type=int, metavar='',default=2)
parser.add_argument('-checkpoint1', type=str, metavar='',required=True,help='checkpoint path 1')
parser.add_argument('-checkpoint2', type=str, metavar='',required=True,help='checkpoint path 2')
parser.add_argument('-name', type=str, metavar='',default='demo')

args = parser.parse_args()

# Load the validation dataset
val_batch_size = 1

if args.factor == 2:  
    val_img_dir = 'dataset/z-axis/factor_2/val'
elif args.factor == 4:
    val_img_dir = 'dataset/z-axis/factor_4/val'
else:
    print('please set the value for factor in [2,4]')
    
val_label_dir = 'dataset/z-axis/label/val'
dir_valdict = ut.create_dictionary(val_img_dir,val_label_dir)
val_datasets = dt.MRIDataset(val_img_dir, val_label_dir, dir_dict = dir_valdict,test=False)
sampler = dt.RdnSampler(val_datasets,val_batch_size,True,classes=val_datasets.classes())
val_dataloaders = torch.utils.data.DataLoader(val_datasets,batch_size = val_batch_size,sampler = sampler,
                    shuffle=False,num_workers=1,pin_memory=False,drop_last=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device available is ',device)

dataiter = iter(val_dataloaders)
images, labels, parsers = dataiter.next()

model1 = load_model(args.checkpoint1)
model2 = load_model(args.checkpoint2)

images = images.to(device)
labels = labels.to(device)
parsers = parsers.to(device)


outputs1 = model1(images)
outputs2 = model2(images)

outputs1 = outputs1.squeeze().detach().to('cpu').numpy()
outputs1 = ut.min_max_normalize(outputs1)
outputs2 = outputs2.squeeze().detach().to('cpu').numpy()
outputs2 = ut.min_max_normalize(outputs2)
images = images.squeeze().to('cpu').numpy()
labels = labels.squeeze().detach().to('cpu').numpy()

# plot images
fnsize = 27
fig = plt.figure(figsize=(20,13))
k=1
for i in range(3):
    fig.add_subplot(1, 4, k)
    if i==0:
        plt.title('Input',fontsize=fnsize)
    plt.imshow(images,cmap='gray')
    psnr = peak_signal_noise_ratio(labels,images)
    ssim = structural_similarity( labels,images, multichannel=False)
    plt.xlabel('PSNR=%.2f\nSSIM=%.4f' % (psnr, ssim),fontsize=25)

    fig.add_subplot(1,4,k+1)
    if i==1:
        plt.title('output for model1',fontsize=fnsize)
    plt.imshow(outputs1,cmap='gray')
    psnr = peak_signal_noise_ratio(labels, outputs1)
    ssim = structural_similarity(labels, outputs1, multichannel=False)
    plt.xlabel('PSNR=%.2f\nSSIM=%.4f' % (psnr, ssim),fontsize=25)

    fig.add_subplot(1,4,k+2)
    # if i==2:
    plt.title('output for model 2',fontsize=fnsize)
    plt.imshow(outputs2,cmap='gray')
    psnr = peak_signal_noise_ratio(labels, outputs2)
    ssim = structural_similarity(labels, outputs2, multichannel=False)
    plt.xlabel('PSNR=%.2f\nSSIM=%.4f' % (psnr, ssim),fontsize=25)

    fig.add_subplot(1,4,k+3)
    plt.title('Labels',fontsize=fnsize)    
    plt.imshow(labels,cmap='gray')
    psnr = peak_signal_noise_ratio(labels, labels)
    ssim = structural_similarity(labels, labels, multichannel=False)
    plt.xlabel('PSNR=%.2f\nSSIM=%.4f' % (psnr, ssim),fontsize=25)
    

plt.tight_layout()
plt.show()


save_name ='compare_'+str(args.name)+'.png'
plt.savefig('samples/'+save_name)
    
   

    
    