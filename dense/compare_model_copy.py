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
import sys

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
parser.add_argument('-name', type=str, metavar='',default='name for saving plot')
parser.add_argument('-model1-name', type=str, metavar='',help='name for model checkpoint1',default='model 1')
parser.add_argument('-model2-name', type=str, metavar='',help='name for model checkpoint2',default='model 2')


args = parser.parse_args()

# Load the validation dataset
val_batch_size = 3

if args.factor == 2:  
      val_img_dir = '../resolution_dataset/dataset/factor_2/val'
elif args.factor == 4:
    print('reached here')
    val_img_dir = '../resolution_dataset/dataset/factor_4/val'
elif args.factor == 8:
    val_img_dir = '../resolution_dataset/dataset/factor_8/val'
else:
    print('please set the value for factor in [2,4,8]')
    
val_label_dir = '../resolution_dataset/dataset/label/val'
dir_valdict = ut.create_dictionary(val_img_dir,val_label_dir)


val_datasets = dt.MRIDataset(val_img_dir, val_label_dir, dir_dict = dir_valdict,test=False)
sampler = dt.RdnSampler(val_datasets,val_batch_size,True,classes=val_datasets.classes())
val_dataloaders = torch.utils.data.DataLoader(val_datasets,batch_size = val_batch_size,sampler = sampler,
                    shuffle=False,num_workers=1,pin_memory=False,drop_last=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device available is ',device)

dataiter = iter(val_dataloaders)
images, labels, parsers = dataiter.next()

#dataloader creates batch of same size images so same type images are plotted whole time
images_lst = []
labels_lst =[]
images_lst.append(images[0,:,:,:])
labels_lst.append(labels[0,:,:,:])
count=1

for i in range(200):
    if count==3:
        break;
    dataiter = iter(val_dataloaders)
    images2, labels2, parsers = dataiter.next()
    if count==1 and images_lst[0].shape != images2[2,:,:,:].shape:
        images_lst.append(images2[2,:,:,:])
        labels_lst.append(labels2[2,:,:,:])
        count +=1
    if count==2  and images_lst[0].shape != images2[0,:,:,:].shape and images_lst[1].shape != images2[0,:,:,:].shape:
        images_lst.append(images2[0,:,:,:])
        labels_lst.append(labels2[0,:,:,:])
        count +=1  

for i in range(1,3):
    if len(images_lst)<3:
        images_lst.append(images[i,:,:,:])
        labels_lst.append(labels[i,:,:,:])


model1 = load_model(args.checkpoint1)
model2 = load_model(args.checkpoint2)

images = images.to(device)
labels = labels.to(device)

outputs1 = model1(images)
outputs2 = model2(images)

def apply_model(model,image,device):
    image=image.unsqueeze(axis=0)
    image=  image.to(device)
    output = model(image)
    return output
   
#plot images
fnsize = 17
fig = plt.figure(figsize=(15,13))
k=1
for i in range(3):

    # input_el = images[i].squeeze().to('cpu').numpy()
    # outputs1_el = outputs1[i].squeeze().detach().to('cpu').numpy()
    # outputs2_el = outputs2[i].squeeze().detach().to('cpu').numpy()
    # label_el = labels[i].squeeze().to('cpu').numpy()

    input_el = images_lst[i].squeeze().to('cpu').numpy()
    outputs1_el = apply_model(model1,images_lst[i],device)
    outputs1_el = outputs1_el.squeeze().detach().to('cpu').numpy()

    outputs2_el = apply_model(model2,images_lst[i],device)
    outputs2_el = outputs2_el.squeeze().detach().to('cpu').numpy()

    label_el = labels_lst[i].squeeze().to('cpu').numpy()

    outputs1_el = ut.min_max_normalize(outputs1_el)
    outputs2_el = ut.min_max_normalize(outputs2_el)
     
    fig.add_subplot(3,4, k)
    if i==0:
        plt.title('Input',fontsize=fnsize)
    plt.imshow(input_el,cmap='gray')
    psnr = peak_signal_noise_ratio(label_el,input_el)
    ssim = structural_similarity( label_el,input_el, multichannel=False)
    plt.xlabel('PSNR=%.2f\nSSIM=%.4f' % (psnr, ssim),fontsize=25)

    
    fig.add_subplot(3,4,k+1)
    if i==0:
        plt.title(str(args.model1_name),fontsize=fnsize)
    plt.imshow(outputs1_el,cmap='gray')
    psnr = peak_signal_noise_ratio(label_el,outputs1_el)
    ssim = structural_similarity( label_el,outputs1_el, multichannel=False)
    plt.xlabel('PSNR=%.2f\nSSIM=%.4f' % (psnr, ssim),fontsize=25)

  
    fig.add_subplot(3,4,k+2)
    if i==0:
        plt.title(str(args.model2_name),fontsize=fnsize)
    plt.imshow(outputs2_el,cmap='gray')
    psnr = peak_signal_noise_ratio(label_el,outputs2_el)
    ssim = structural_similarity( label_el,outputs2_el, multichannel=False)
    plt.xlabel('PSNR=%.2f\nSSIM=%.4f' % (psnr, ssim),fontsize=25)

    fig.add_subplot(3,4,k+3)
    if i==0:
        plt.title('Labels',fontsize=fnsize)  
    plt.imshow(label_el,cmap='gray')
    psnr = peak_signal_noise_ratio(label_el,label_el)
    ssim = structural_similarity( label_el,label_el, multichannel=False)
    plt.xlabel('PSNR=%.2f\nSSIM=%.4f' % (psnr, ssim),fontsize=25)

    k = k+4
  
plt.tight_layout()
plt.show()


save_name ='compare_'+str(args.name)+'.png'
plt.savefig('samples/'+save_name)
    
   

    
    