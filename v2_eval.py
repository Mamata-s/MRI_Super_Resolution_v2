import torch
import torch.nn as nn
import model_concat as mdl
import model_utility as ut
import measure_error as err

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import warnings
warnings.filterwarnings("ignore")
import dataset as dt


# Load the validation dataset

val_batch_size = 1
val_img_dir = 'resolution_dataset/val_8_full/input'
val_label_dir = 'resolution_dataset/val_8_full/label'

dir_valdict = ut.create_dictionary(val_img_dir,val_label_dir)

val_datasets = dt.MRIDataset(val_img_dir, val_label_dir, dir_dict = dir_valdict,test=False)
sampler = dt.RdnSampler(val_datasets,val_batch_size,True,classes=val_datasets.classes())
val_dataloaders = torch.utils.data.DataLoader(val_datasets,batch_size = val_batch_size,sampler = sampler,
                    shuffle=False,num_workers=1,pin_memory=False,drop_last=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print('device available is ',device)


model = mdl.MRINetV2(device,mode='eval')
model_save_name = 'model_v2_concat__final.pth'
path = F"model/concat/{model_save_name}"
print('eval mrinet v2 network')

model = nn.DataParallel(model,device_ids=[0,1,2,3])
model.to(device)
model.load_state_dict(torch.load(path))

mse_loss = nn.MSELoss()

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
    l1 = err.calculate_l1_distance(out,lbl)

    l2 = mse_loss(outputs,labels)
    psnr_val = peak_signal_noise_ratio(out,lbl)
    ssim_val = structural_similarity(out,lbl,multichannel=False)
    l1 = err.calculate_l1_distance(out,lbl)
    
    l1_error.append(l1)
    l2_error.append(l2.item())
    psnr.append(psnr_val)
    ssim.append(ssim_val)


print('PSNR for Network',err.Average(psnr))
print('SSIM for Network',err.Average(ssim))
print('L1 ERROR for Network',err.Average(l1_error))
print('MSE for Network',err.Average(l2_error))

    