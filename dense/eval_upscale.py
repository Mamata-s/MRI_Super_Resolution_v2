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
from densenet import SRDenseNet, SRDenseNetUpscale
from utils import AverageMeter, calc_psnr,create_dictionary

parser = argparse.ArgumentParser(description='model demo')
parser.add_argument('-tfactor', type=int, metavar='',required=True,help='trained resolution factor')
parser.add_argument('-factor', type=int, metavar='',required=True,help='resolution factor')
parser.add_argument('-checkpoint', type=str, metavar='',required=True,help='checkpoint path')
args = parser.parse_args()

trained_factor = args.tfactor

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
# print('device available is ',device)

model = SRDenseNetUpscale(num_channels=1, growth_rate=4, num_blocks = 4, num_layers=3).to(device)
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

for idx, (images, labels) in enumerate(val_dataloaders):
    images = images.to(device)
    labels = labels.to(device)

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

    