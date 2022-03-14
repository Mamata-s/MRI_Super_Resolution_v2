import torch
import torch.nn as nn
import  cv2
import model as md
import utils as ut
import dataset as dt
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import dataset as dt
import numpy as np
import argparse
import torch.backends.cudnn as cudnn

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='model demo')
parser.add_argument('-model', type=str, metavar='',required=True,help='name of the model')
parser.add_argument('-tfactor', type=int, metavar='',required=True,help='trained factor')
parser.add_argument('-factor', type=int, metavar='',required=True,help='resolution factor')
parser.add_argument('-checkpoint', type=str, metavar='',required=True,help='checkpoint path')
parser.add_argument('-axis', type=str, metavar='',default=0)
args = parser.parse_args()


# Load the validation dataset
val_batch_size = 1


datapath= 'dataset/f1_160/mag_sos_wn.nii'
data_label = dt.load_data_nii(datapath)

x_max,y_max,z_max = data_label.shape
data_image = dt.preprocess_data(data_label,factor=args.factor,pad=True)


val_datasets = dt.MRI3DDataset(image_arr=data_image, label_arr=data_label,eval=True)


val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size = val_batch_size,shuffle=False,
        num_workers=0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device available is ',device)

dataiter = iter(val_dataloader)
images, labels = dataiter.next()

print('input cube size is ',images.shape)
print('label cube size is ',images.shape)


cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print('device availabe is ',device)

model = md.SR3DDenseNet()
print('training 3DSRDenseNet network')

model = nn.DataParallel(model,device_ids=[0,1,2,3])
model.to(device)
# print(model)

state_dict = model.state_dict()
for n, p in torch.load(args.checkpoint, map_location=lambda storage, loc: storage).items():
    if n in state_dict.keys():
        state_dict[n].copy_(p)
    else:
        raise KeyError(n)

model.eval()

images = images.to(device)
labels = labels.to(device)
images = images.type(torch.cuda.FloatTensor)
labels = labels.type(torch.cuda.FloatTensor)
        
        
outputs = model(images)
print('output shape is ',outputs.shape)



outputs = outputs.squeeze().detach().to('cpu').numpy()
# outputs = ut.min_max_normalize(outputs)
print('output shape after squeeze', outputs.shape)
images = images.squeeze().to('cpu').numpy()
labels = labels.squeeze().detach().to('cpu').numpy()


if (args.axis ==0):
    images = images[45,:,:]
    labels = labels[45,:,:]
    outputs = outputs[45,:,:]
elif(args.axis==1):
    images = images[:,45,:]
    labels = labels[:,45,:]
    outputs = outputs[:,45,:]
else:
    images = images[:,:,45]
    labels = labels[:,:,45]
    outputs = outputs[:,:,45]


# outputs=ut.min_max_normalize(outputs)
# plot images
fnsize = 27
fig = plt.figure(figsize=(9,15))


fig.add_subplot(3, 3, 1)
plt.title('label',fontsize=fnsize)
plt.imshow(labels,cmap='gray')
psnr = peak_signal_noise_ratio(labels, labels)
ssim = structural_similarity(labels, labels, multichannel=False)
plt.xlabel('PSNR=%.2f\nSSIM=%.4f \n (1)' % (psnr, ssim,),fontsize=25)

fig.add_subplot(3, 3, 2)
plt.title('Input',fontsize=fnsize)
plt.imshow(images,cmap='gray')
psnr = peak_signal_noise_ratio(labels, images)
ssim = structural_similarity(labels, images, multichannel=False)
plt.xlabel('PSNR=%.2f\nSSIM=%.4f \n (2)' % (psnr, ssim),fontsize=25)

fig.add_subplot(3, 3, 3)
plt.title('output',fontsize=fnsize)
plt.imshow(outputs,cmap='gray')
psnr = peak_signal_noise_ratio(labels, outputs)
ssim = structural_similarity(labels, outputs, multichannel=False)
plt.xlabel('PSNR=%.2f\nSSIM=%.4f \n (3)' % (psnr, ssim),fontsize=fnsize)


plt.tight_layout()
plt.show()

save_name = str(args.model)+'_t'+ str(args.tfactor)+'_v'+str(args.factor)+'_ax'+str(args.axis)+'.png'

plt.savefig('result/'+save_name)
print('figure saved')
    
   

    
    