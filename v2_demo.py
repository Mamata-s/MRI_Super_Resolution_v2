import torch
import torch.nn as nn
import  cv2
import model_concat as mconcat
import model_utility as ut
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import dataset as dt

import warnings
warnings.filterwarnings("ignore")



# Load the validation dataset
val_batch_size = 1
val_img_dir = 'resolution_dataset/val_8_full/input'
val_label_dir = 'resolution_dataset/val_8_full/label'

# val_img_dir = 'resolution_dataset/val_4/input'
# val_label_dir = 'resolution_dataset/val_4/label'


dir_valdict = ut.create_dictionary(val_img_dir,val_label_dir)
val_datasets = dt.MRIDataset(val_img_dir, val_label_dir, dir_dict = dir_valdict)
sampler = dt.RdnSampler(val_datasets,val_batch_size,True,classes=val_datasets.classes())

val_dataloaders = torch.utils.data.DataLoader(val_datasets, batch_size = val_batch_size, sampler = sampler,shuffle=False,
    num_workers=1,pin_memory=False,drop_last=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device available is ',device)

dataiter = iter(val_dataloaders)
images, labels,parsers = dataiter.next()

model_concat = mconcat.MRINetV2(device,mode='eval')
model_concat = nn.DataParallel(model_concat)



model_save_name = 'model_v2_concat__final.pth'
path = F"model/concat/{model_save_name}"
model_concat.load_state_dict(torch.load(path))

model_concat.eval()

images = images.to(device)
labels = labels.to(device)
parsers = parsers.to(device)

outputs = model_concat(images)
out = outputs
out = out.squeeze().detach().to('cpu').numpy()
out = ut.min_max_normalize(out)
images = images.squeeze().to('cpu').numpy()
labels = labels.squeeze().detach().to('cpu').numpy()

# plot images
fig = plt.figure(figsize=(10,3))

fig.add_subplot(1, 5, 1)
plt.title('Input')
plt.imshow(images,cmap='gray', vmin=0, vmax=1)

fig.add_subplot(1, 5, 2)
plt.title('label')
plt.imshow(labels,cmap='gray', vmin=0, vmax=1)
psnr = peak_signal_noise_ratio(images, labels)
ssim = structural_similarity(images, labels, multichannel=False)
# plt.xlabel('PSNR=%.2f\nSSIM=%.4f' % (psnr, ssim))

fig.add_subplot(1, 5, 3)
plt.title('output')
plt.imshow(out,cmap='gray', vmin=0, vmax=1)
psnr = peak_signal_noise_ratio(images, out)
ssim = structural_similarity(images, out, multichannel=False)
# plt.xlabel('PSNR=%.2f\nSSIM=%.4f' % (psnr, ssim))

fig.add_subplot(1, 5, 4)
plt.title('Error: label-input')
# error = np.abs(labels-images)
error = (labels-images)
plt.imshow(error*100,cmap='gray', vmin=0, vmax=1)


fig.add_subplot(1, 5, 5)
plt.title('Error : label-output')
# error = np.abs(labels-out)
error = (labels-out)
plt.imshow(error*100,cmap='gray', vmin=0, vmax=1)

plt.tight_layout()
plt.show()


plt.savefig('result_mrinet_v2_full.png')

    
   

    
    