import torch
import torch.nn as nn
import model_concat as mdl
import model_utility as ut
import measure_error as err

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import warnings
warnings.filterwarnings("ignore")
import argparse
import dataset as dt
import argparse

parser = argparse.ArgumentParser(description='model demo')
parser.add_argument('-model', type=str, metavar='',required=True,help='name of the model')
parser.add_argument('-tfactor', type=int, metavar='',required=True,help='trained resolution factor')
parser.add_argument('-factor', type=int, metavar='',required=True,help='resolution factor')
parser.add_argument('-checkpoint', type=str, metavar='',required=True,help='checkpoint path')
parser.add_argument('--bicubic',
                    help='use dataset from bicubic upsampling', action='store_true')

args = parser.parse_args()

trained_factor = args.tfactor
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
# print('device available is ',device)

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
state_dict = model.state_dict()
for n, p in torch.load(args.checkpoint, map_location=lambda storage, loc: storage).items():
    if n in state_dict.keys():
        state_dict[n].copy_(p)
    else:
        raise KeyError(n)


# checkpoint = torch.load(path)
# model.load_state_dict(checkpoint['model_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']
# trained_factor = checkpoint['factor']



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
    if args.model == 'mrinet':
        outputs = outputs[0]
    out = outputs.squeeze().detach().to('cpu').numpy()
    out = ut.min_max_normalize(out)
    img = images.squeeze().detach().to('cpu').numpy()
    lbl = labels.squeeze().detach().to('cpu').numpy()

    # measure error
    if args.model == 'initial':
      l2 = mse_loss(labels,images)
      psnr_val = peak_signal_noise_ratio(lbl,img)
      ssim_val = structural_similarity(lbl,img,multichannel=False)
      l1 = l1_loss(labels,images)
    else:
      l2 = mse_loss(labels, outputs)
      psnr_val = peak_signal_noise_ratio(lbl, out)
      ssim_val = structural_similarity(lbl,out,multichannel=False)
      l1 = l1_loss(labels,outputs)
    
    l1_error.append(l1.item())
    l2_error.append(l2.item())
    psnr.append(psnr_val)
    ssim.append(ssim_val)


print ('****** Model info *****')
print("Model trained on dataset factor %d  and evaluated on factor %d" % (args.tfactor,args.factor))

if args.model == 'initial':
  print('PSNR for Initial(input and label)',err.Average(psnr))
  print('SSIM for Initial(input and label)',err.Average(ssim))
  print('L1 ERROR for Initial(input and label)',err.Average(l1_error))
  print('MSE for Initial(input and label)',err.Average(l2_error))
else:
  print('PSNR for Network',err.Average(psnr))
  print('SSIM for Network',err.Average(ssim))
  print('L1 ERROR for Network',err.Average(l1_error))
  print('MSE for Network',err.Average(l2_error))

    