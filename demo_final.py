import torch
import torch.nn as nn
import  cv2
import model_concat as mdl
import model_utility as ut
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import dataset as dt
import numpy as np
import argparse

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='model demo')
parser.add_argument('-model', type=str, metavar='',required=True,help='name of the model')
parser.add_argument('-tfactor', type=int, metavar='',required=True,help='trained factor')
parser.add_argument('-factor', type=int, metavar='',required=True,help='resolution factor')
parser.add_argument('-checkpoint', type=str, metavar='',required=True,help='checkpoint path')
parser.add_argument('--bicubic',
                    help='use dataset from bicubic upsampling', action='store_true')
args = parser.parse_args()


# Load the validation dataset
val_batch_size = 1

if args.bicubic:
  print('evaluating on bicubic images')
  if args.factor == 2:  
      val_img_dir = 'resolution_dataset/dataset/crop_bicubic_factor_2/val'
  elif args.factor == 4:
      print('reached here')
      val_img_dir = 'resolution_dataset/dataset/crop_bicubic_factor_4/val'
  elif args.factor == 8:
      val_img_dir = 'resolution_dataset/dataset/crop_bicubic_factor_8/val'
  else:
      print('please set the value for factor in [2,4,8]')
else:
  if args.factor == 2:  
      val_img_dir = 'resolution_dataset/dataset/factor_2/val'
  elif args.factor == 4:
      print('reached here')
      val_img_dir = 'resolution_dataset/dataset/factor_4/val'
  elif args.factor == 8:
      val_img_dir = 'resolution_dataset/dataset/factor_8/val'
  else:
      print('please set the value for factor in [2,4,8]')
    
val_label_dir = 'resolution_dataset/dataset/label/val'
dir_valdict = ut.create_dictionary(val_img_dir,val_label_dir)
val_datasets = dt.MRIDataset(val_img_dir, val_label_dir, dir_dict = dir_valdict,test=False)
sampler = dt.RdnSampler(val_datasets,val_batch_size,True,classes=val_datasets.classes())
val_dataloaders = torch.utils.data.DataLoader(val_datasets,batch_size = val_batch_size,sampler = sampler,
                    shuffle=False,num_workers=1,pin_memory=False,drop_last=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device available is ',device)

dataiter = iter(val_dataloaders)
images, labels,parsers = dataiter.next()

# try:
#   images, labels,parsers = dataiter.next()
# except StopIteration:
#   dataiter = iter(val_dataloaders)
#   images, labels,parsers = dataiter.next()

if args.model == 'coarser':
  model = mdl.CoarseNetwork()
  print('eval coarser network')
elif args.model == 'shuffle':
  model = mdl.SRshuffle(1)
  print('eval shuffle network')
elif args.model == 'srcnn':
  model = mdl.SRCNN()
  print('eval srcnn network')
elif args.model == 'srnet':
  model = mdl.SRNet()
  print('eval srnet ')
elif args.model == 'mrinet':
  model = mdl.MRINet(device)
  print('eval mrinet ')
elif args.model=='initial':
  model = mdl.SRCNN()
  print('eval for initial condition')
else:
    print('please select valid model')

model = nn.DataParallel(model,device_ids=[0,1,2,3])
model.to(device)

# checkpoint = torch.load(args.checkpoint)
# model.load_state_dict(checkpoint['model_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']
# trained_factor = checkpoint['factor']

state_dict = model.state_dict()
for n, p in torch.load(args.checkpoint, map_location=lambda storage, loc: storage).items():
    if n in state_dict.keys():
        state_dict[n].copy_(p)
    else:
        raise KeyError(n)

model.eval()

images = images.to(device)
labels = labels.to(device)
parsers = parsers.to(device)

outputs = model(images)
if args.model=='mrinet':
    out = outputs[0]
else:
    out = outputs
out = out.squeeze().detach().to('cpu').numpy()
out = ut.min_max_normalize(out)
images = images.squeeze().to('cpu').numpy()
labels = labels.squeeze().detach().to('cpu').numpy()

# plot images
fnsize = 27
fig = plt.figure(figsize=(20,13))

fig.add_subplot(1, 5, 1)
plt.title('Input',fontsize=fnsize)
plt.imshow(images,cmap='gray')
psnr = peak_signal_noise_ratio(labels, images)
ssim = structural_similarity(labels, images, multichannel=False)
plt.xlabel('PSNR=%.2f\nSSIM=%.4f' % (psnr, ssim),fontsize=25)

fig.add_subplot(1, 5, 2)
plt.title('label',fontsize=fnsize)
plt.imshow(labels,cmap='gray')
psnr = peak_signal_noise_ratio(labels, labels)
ssim = structural_similarity(labels, labels, multichannel=False)
plt.xlabel('PSNR=%.2f\nSSIM=%.4f' % (psnr, ssim),fontsize=25)


fig.add_subplot(1, 5, 3)
plt.title('output',fontsize=fnsize)
plt.imshow(out,cmap='gray')
psnr = peak_signal_noise_ratio(labels, out)
ssim = structural_similarity(labels, out, multichannel=False)
plt.xlabel('PSNR=%.2f\nSSIM=%.4f' % (psnr, ssim),fontsize=fnsize)

fig.add_subplot(1, 5, 4)
plt.title('Error: label-input')
error = np.abs(labels-images)
# error = (labels-images)
plt.imshow(error,cmap='gray')
# plt.imshow(error*100,cmap='gray', vmin=0, vmax=1)


fig.add_subplot(1, 5, 5)
plt.title('Error : label-output')
error = np.abs(labels-out)
# error = (labels-out)
plt.imshow(error,cmap='gray')
# plt.imshow(error*100,cmap='gray', vmin=0, vmax=1)

plt.tight_layout()
plt.show()

if args.bicubic:
  save_name = str(args.model)+'_bicubic_t'+ str(args.tfactor)+'_v'+str(args.factor)+'.png'
else:
  save_name = str(args.model)+'_t'+ str(args.tfactor)+'_v'+str(args.factor)+'.png'

plt.savefig('result/'+save_name)
print('figure saved')
    
   

    
    