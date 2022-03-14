import torch
import torch.nn as nn
import utils as ut

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import warnings
warnings.filterwarnings("ignore")
import argparse
import dataset as dt
import argparse
from densenet import SRDenseNet

parser = argparse.ArgumentParser(description='model demo')
parser.add_argument('-tfactor', type=int, metavar='',required=True,help='trained resolution factor')
parser.add_argument('-factor', type=int, metavar='',required=True,help='resolution factor')
parser.add_argument('-checkpoint', type=str, metavar='',required=True,help='checkpoint path')
parser.add_argument('--bicubic',
                    help='use dataset from bicubic upsampling', action='store_true')
args = parser.parse_args()

trained_factor = args.tfactor

val_batch_size=1
# Load the validation dataset
if args.bicubic:
  print('evaluating on bicubic images')
  if args.factor == 2:  
      val_img_dir = '../resolution_dataset/dataset/crop_bicubic_factor_2/val'
  elif args.factor == 4:
      print('reached here')
      val_img_dir = '../resolution_dataset/dataset/crop_bicubic_factor_4/val'
  elif args.factor == 8:
      val_img_dir = '../resolution_dataset/dataset/crop_bicubic_factor_8/val'
  else:
      print('please set the value for factor in [2,4,8]')
else:
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
# print('device available is ',device)

model = SRDenseNet(num_channels=1, growth_rate=4, num_blocks = 4, num_layers=3).to(device)
print('eval SRDenseNet network')


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


mse_loss = nn.MSELoss()
l1_loss = nn.L1Loss()

model.eval()
l1_error =[]
l2_error = []
psnr =[]
ssim=[]

for idx, (images, labels,parsers) in enumerate(val_dataloaders):
    images = images.to(device)
    labels = labels.to(device)
    parsers = parsers.to(device)

    outputs = model(images)
    out = outputs.squeeze().detach().to('cpu').numpy()
    out = ut.min_max_normalize(out)
    img = images.squeeze().detach().to('cpu').numpy()
    lbl = labels.squeeze().detach().to('cpu').numpy()

    # measure error
    
    l2 = mse_loss(outputs,labels)
    psnr_val = peak_signal_noise_ratio(out,lbl)
    ssim_val = structural_similarity(out,lbl,multichannel=False)
    l1 = l1_loss(outputs,labels)
    
    l1_error.append(l1.item())
    l2_error.append(l2.item())
    psnr.append(psnr_val)
    ssim.append(ssim_val)


print ('****** Model info *****')
print("Model trained on dataset factor %d  and evaluated on factor %d" % (args.tfactor,args.factor))


print('PSNR for Network',ut.Average(psnr))
print('SSIM for Network',ut.Average(ssim))
print('L1 ERROR for Network',ut.Average(l1_error))
print('MSE for Network',ut.Average(l2_error))

    