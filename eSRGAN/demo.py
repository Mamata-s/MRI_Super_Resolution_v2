import torch
import torch.nn as nn
import  cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from utils import create_dictionary,min_max_normalize
from dataset import MRIDataset, RdnSampler
from model import Discriminator, Generator, ContentLoss


import warnings
warnings.filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser(description='model demo')
parser.add_argument('-tfactor', type=int, metavar='',required=True,help='trained factor')
parser.add_argument('-factor', type=int, metavar='',required=True,help='resolution factor')
parser.add_argument('-checkpoint', type=str, metavar='',required=True,help='checkpoint path')
args = parser.parse_args()


# Load the validation dataset

val_batch_size = 1
if args.factor == 2:  
    val_img_dir = '../resolution_dataset/dataset/factor_2/val'
elif args.factor == 4:
    val_img_dir = '../resolution_dataset/dataset/factor_4/val'
elif args.factor == 8:
    val_img_dir = '../resolution_dataset/dataset/factor_8/val'
else:
    print('please set the value for factor in [2,4,8]')


val_label_dir = '../resolution_dataset/dataset/label/val'
dir_valdict = create_dictionary(val_img_dir,val_label_dir)
val_datasets = MRIDataset(val_img_dir, val_label_dir, dir_dict = dir_valdict,test=False)
sampler = RdnSampler(val_datasets,val_batch_size,True,classes=val_datasets.classes())
val_dataloaders = torch.utils.data.DataLoader(val_datasets,batch_size = val_batch_size,sampler = sampler,
                    shuffle=False,num_workers=1,pin_memory=False,drop_last=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device available is ',device)

dataiter = iter(val_dataloaders)
images, labels,parsers = dataiter.next()


model = Generator().to(device)
path = args.checkpoint
print('eval SRDENSE network')

state_dict = model.state_dict()
for n, p in torch.load(path, map_location=lambda storage, loc: storage).items():
    if n in state_dict.keys():
        state_dict[n].copy_(p)
    else:
        raise KeyError(n)

model.to(device)
model.eval()

images = images.to(device)
labels = labels.to(device)
parsers = parsers.to(device)

outputs = model(images) 

out = outputs
out = out.squeeze().detach().to('cpu').numpy()
out = min_max_normalize(out)
images = images.squeeze().to('cpu').numpy()
labels = labels.squeeze().detach().to('cpu').numpy()

# plot images
fnsize = 27
fig = plt.figure(figsize=(20,13))

fig.add_subplot(1, 3, 1)
plt.title('Input',fontsize=fnsize)
plt.imshow(images,cmap='gray')
psnr = peak_signal_noise_ratio(images, images)
ssim = structural_similarity(images, images, multichannel=False)
plt.xlabel('PSNR=%.2f\nSSIM=%.4f' % (psnr, ssim))

fig.add_subplot(1, 3, 2)
plt.title('label',fontsize=fnsize)
plt.imshow(labels,cmap='gray')
psnr = peak_signal_noise_ratio(images, labels)
ssim = structural_similarity(images, labels, multichannel=False)
plt.xlabel('PSNR=%.2f\nSSIM=%.4f' % (psnr, ssim),fontsize=fnsize)

fig.add_subplot(1, 3, 3)
plt.title('output')
plt.imshow(out,cmap='gray')
psnr = peak_signal_noise_ratio(images, out)
ssim = structural_similarity(images, out, multichannel=False)
plt.xlabel('PSNR=%.2f\nSSIM=%.4f' % (psnr, ssim),fontsize=fnsize)

# fig.add_subplot(1, 5, 4)
# plt.title('Error: label-input')
# error = np.abs(labels-images)
# plt.imshow(error*100,cmap='gray')

# fig.add_subplot(1, 5, 5)
# plt.title('Error : label-output')
# error = np.abs(labels-out)
# plt.imshow(error*100,cmap='gray')

plt.tight_layout()
plt.show()

save_name = 'samples/GAN'+'_t'+str(args.tfactor)+'_v'+str(args.factor)+'.png'
plt.savefig(save_name)