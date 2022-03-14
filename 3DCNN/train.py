
import torch
import torch.nn as nn
import  cv2
import time
import utils as ut
import dataset as dt
import os
import copy
import model as md
from torch.cuda import amp

import warnings
warnings.filterwarnings("ignore")
import argparse
import torch.backends.cudnn as cudnn
import pickle


# print(torch.cuda.device_count())

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    lr = args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

parser = argparse.ArgumentParser(description='model demo')
parser.add_argument('-factor', type=int, metavar='',required=False,help='resolution factor',default=2)
parser.add_argument('-epoch', type=int, metavar='',required=False,help='epochs number',default=500)
parser.add_argument('-lr', type=float, metavar='',required=False,help='learning rate',default=0.00001)
parser.add_argument('-batch_size', type=int, metavar='',required=False,help='batch size',default=3)
args = parser.parse_args()

train_batch_size = args.batch_size
val_batch_size =1
cube_size= 98


datapath= 'dataset/f1_160/mag_sos_wn.nii'
data_label = dt.load_data_nii(datapath)

x_max,y_max,z_max = data_label.shape
data_image = dt.preprocess_data(data_label,factor=args.factor,pad=True)

index_list = dt.create_index(120, cube_size, x_max, y_max, z_max)
val_index_list = dt.create_index(60, cube_size, x_max, y_max, z_max)


train_datasets = dt.MRI3DDataset(image_arr=data_image, label_arr=data_label,indexes=index_list,cube_size=cube_size)

val_datasets =  dt.MRI3DDataset(image_arr=data_image,label_arr=data_label,indexes=val_index_list,cube_size=cube_size)

train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size = train_batch_size,shuffle=True,
        num_workers=0)

eval_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size = val_batch_size,shuffle=False,
        num_workers=0)

cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print('device availabe is ',device)

model = md.SR3DDenseNet()
print('training 3DSRDenseNet network')

model = nn.DataParallel(model,device_ids=[0,1,2,3])
model.to(device)
# print(model)

optim = torch.optim.SGD(params = model.parameters(), lr = args.lr, momentum=0.9)

n_epochs= args.epoch
output_dir = 'model/'
path = f'model/mse_error/3DSRDenseNet'+'_final.pth'


print('*********** Begin Training *************')



model.train() 
scaler = amp.GradScaler()
losses = []
k = 0
alpha = 0.003
beta = 0.02
start = time.time()
best_epoch = 0
best_psnr = 0.0
mse_loss = nn.MSELoss()
l1_loss = nn.L1Loss()

for epoch in range(1, n_epochs+1):
    adjust_learning_rate(optim, epoch)
    model.train()
    epoch_losses = ut.AverageMeter()
    for idx, (images, labels) in enumerate(train_dataloaders):
        images = images.to(device)
        labels = labels.to(device)
        
    
        images = images.type(torch.cuda.FloatTensor)
        labels = labels.type(torch.cuda.FloatTensor)
        
    
        # print('device of the images', images.is_cuda)
        with torch.cuda.amp.autocast():
            outputs = model(images)
            outputs = (outputs-outputs.min())/(outputs.max()-outputs.min())
            loss = mse_loss(outputs, labels)
            # loss = l1_loss(outputs, labels) 

        # print('upto output from model')
        # print('input max :', images.max())
        # print('input min :', images.min())
        # print('output max :', outputs.max())
        # print('output min :', outputs.min())
    
        # print('output tensor type :', outputs.type())
        # print('input tensor type :', images.type())
        # print('labels tensor type :', labels.type())

        optim.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        losses.append(float(loss))
        
        if idx % 100 == 0:
            print("Epoch [%d/%d]. Iter [%d/%d]. Loss: %0.2f" % (epoch, n_epochs, idx + 1, len(train_dataloaders), loss))
    if k == 100:
        torch.save(model.state_dict(), os.path.join('model/mse_error', '3d_densenet_epoch_{}_f_{}.pth'.format(epoch,args.factor)))
        k=0
    k+=1
    # model.eval()
    # epoch_psnr = ut.AverageMeter()
    # for idx, (images, labels) in enumerate(eval_dataloader):
    #   images = images.to(device)
    #   labels = labels.to(device)

    #   images = images.type(torch.cuda.FloatTensor)
    #   labels = labels.type(torch.cuda.FloatTensor)

    #   with torch.no_grad():
    #       preds = model(images).clamp(0.0, 1.0)

    #   epoch_psnr.update(ut.calc_psnr(preds.squeeze().detach().to('cpu').numpy(), labels.squeeze().detach().to('cpu').numpy()), len(images))

    # print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

    # if epoch_psnr.avg > best_psnr:
    #     best_epoch = epoch
    #     best_psnr = epoch_psnr.avg
    #     best_weights = copy.deepcopy(model.state_dict())
      
# print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
torch.save(model.state_dict(), os.path.join(path))

print('************ Training completed ******************')
end = time.time()
print('total time taken for training is', (end-start))

with open("losses", "wb") as fp:   #Pickling
    pickle.dump(losses, fp)