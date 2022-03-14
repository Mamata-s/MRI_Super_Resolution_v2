import torch
import torch.nn as nn
import  cv2
import time
import model_utility as ut
import model_concat as mdl
import measure_error as err
import dataset as dt
import os
import copy

import warnings
warnings.filterwarnings("ignore")
import argparse

# print(torch.cuda.device_count())

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    lr = args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

parser = argparse.ArgumentParser(description='model demo')
parser.add_argument('-model', type=str, metavar='',required=True,help='name of the model')
parser.add_argument('-factor', type=int, metavar='',required=False,help='resolution factor',default=2)
parser.add_argument('-epoch', type=int, metavar='',required=False,help='epochs number',default=500)
parser.add_argument('-lr', type=float, metavar='',required=False,help='learning rate',default=0.000001)
parser.add_argument('-batch_size', type=int, metavar='',required=False,help='batch size',default=32)
parser.add_argument('--bicubic', help='use dataset from bicubic upsampling', action='store_true')

args = parser.parse_args()




train_batch_size = args.batch_size


if args.bicubic:
  str_train_2 = 'crop_bicubic_factor_2/train'
  str_train_4 = 'crop_bicubic_factor_4/train'
  str_train_8 = 'crop_bicubic_factor_8/train'
  str_val_2 = 'crop_bicubic_factor_2/val'
  str_val_4 = 'crop_bicubic_factor_4/val'
  str_val_8 = 'crop_bicubic_factor_8/val'
  print('training from bicubic upsampled images')
        
else:
  str_train_2 = 'factor_2/train'
  str_train_4 = 'factor_4/train'
  str_train_8 = 'factor_8/train'
  str_val_2 = 'factor_2/val'
  str_val_4 = 'factor_4/val'
  str_val_8 = 'factor_8/val'

if args.factor == 2:  
  train_img_dir = 'resolution_dataset/dataset/'+ str(str_train_2)
  val_img_dir = 'resolution_dataset/dataset/'+str(str_val_2)
elif args.factor == 4:
  train_img_dir = 'resolution_dataset/dataset/'+str(str_train_4)
  val_img_dir = 'resolution_dataset/dataset/'+ str(str_val_4)
elif args.factor == 8:
  train_img_dir = 'resolution_dataset/dataset/'+ str(str_train_8)
  val_img_dir = 'resolution_dataset/dataset/'+str(str_val_8)
else:
    print('please set the value for factor in [2,4,8]')

train_label_dir = 'resolution_dataset/dataset/label/train'
dir_traindict = ut.create_dictionary(train_img_dir,train_label_dir)
train_datasets = dt.MRIDataset(train_img_dir, train_label_dir, dir_dict = dir_traindict,test=False)
sampler = dt.RdnSampler(train_datasets,train_batch_size,True,classes=train_datasets.classes())

val_batch_size =1
val_label_dir = 'resolution_dataset/dataset/label/val'
val_dir_traindict = ut.create_dictionary(val_img_dir,val_label_dir)
val_datasets = dt.MRIDataset(val_img_dir, val_label_dir, dir_dict = val_dir_traindict,test=True)
val_sampler = dt.RdnSampler(val_datasets,val_batch_size,True,classes=train_datasets.classes())


eval_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size = val_batch_size,sampler = val_sampler,shuffle=False,
        num_workers=1,pin_memory=False,drop_last=False)

train_dataloaders = torch.utils.data.DataLoader(
    train_datasets,
    batch_size = train_batch_size,        
    sampler = sampler,
    shuffle=False,
    num_workers=1,
    pin_memory=False,
    drop_last=False)


mse_loss = nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device availabe is ',device)


if args.model == 'coarser':
  model = mdl.CoarseNetwork()
  print('training coarser network')
elif args.model== 'shuffle':
  model = mdl.SRshuffle(1)
  print('training Shuffle network')
elif args.model== 'srnet':
  model = mdl.SRNet()
  print('training SRNet ')
elif args.model== 'mrinet':
  model = mdl.MRINet(device)
  print('training MRINet ')
else:
  model = mdl.SRCNN()
  print('training srcnn network')


model = nn.DataParallel(model,device_ids=[0,1,2,3])
model.to(device)
# print(model)


optim = torch.optim.SGD(params = model.parameters(), lr = args.lr, momentum=0.9)

n_epochs= args.epoch
if args.bicubic:
  if args.model == 'coarser':
    path = f'model/coarser_bicubic_e'+str(n_epochs)+'_f'+str(args.factor)+'_final.pth'
  elif args.model== 'shuffle':
    path = f'model/shuffle_bicubic_e'+str(n_epochs)+'_f'+str(args.factor)+'_final.pth'
  elif args.model== 'srnet':
    path = f'model/srnet_bicubic_e'+str(n_epochs)+'_f'+str(args.factor)+'_final.pth'
  elif args.model== 'mrinet':
    path = f'model/mrinet_bicubic_e'+str(n_epochs)+'_f'+str(args.factor)+'_final.pth'
  else:
    path = f'model/srcnn_bicubic_e'+str(n_epochs)+'_f'+str(args.factor)+'_final.pth'
else:
  if args.model == 'coarser':
    path = f'model/coarser_e'+str(n_epochs)+'_f'+str(args.factor)+'_final.pth'
  elif args.model== 'shuffle':
    path = f'model/shuffle_e'+str(n_epochs)+'_f'+str(args.factor)+'_final.pth'
  elif args.model== 'srnet':
    path = f'model/srnet_e'+str(n_epochs)+'_f'+str(args.factor)+'_final.pth'
  elif args.model== 'mrinet':
    path = f'model/mrinet_e'+str(n_epochs)+'_f'+str(args.factor)+'_final.pth'
  else:
    path = f'model/srcnn_e'+str(n_epochs)+'_f'+str(args.factor)+'_final.pth'

print('*********** Begin Training *************')

model.train() 
losses = []
k = 0
alpha = 0.003
beta = 0.02
start = time.time()
best_epoch = 0
best_psnr = 0.0
for epoch in range(1, n_epochs+1):
    adjust_learning_rate(optim, epoch)
    model.train()
    epoch_losses = err.AverageMeter()
    for idx, (images, labels,parsers) in enumerate(train_dataloaders):
        images = images.to(device)
        labels = labels.to(device)
        parsers = parsers.to(device)
        torch.cuda.empty_cache()
        if args.model=='mrinet':
          outputs,f,p,y_c = model(images)
          loss_coarser = mse_loss(y_c, labels) 
          loss_decoder = mse_loss(outputs, labels) 
          loss_parser = mse_loss(p, parsers) 
          loss = (loss_coarser*beta) + (alpha * loss_decoder) + (loss_parser * (1-alpha))
        else:
          outputs = model(images)
          loss = mse_loss(outputs, labels)  
        epoch_losses.update(loss.item(), len(images))
        optim.zero_grad()
        loss.backward()
        optim.step() 
        losses.append(float(loss))
        
        if idx % 100 == 0:
            print("Epoch [%d/%d]. Iter [%d/%d]. Loss: %0.2f" % (epoch, n_epochs, idx + 1, len(train_dataloaders), loss))
    model.eval()
    epoch_psnr = err.AverageMeter()
    for idx, (images, labels,parsers) in enumerate(eval_dataloader):
      images = images.to(device)
      labels = labels.to(device)
      parsers = parsers.to(device)

      with torch.no_grad():
          preds = model(images).clamp(0.0, 1.0)
      epoch_psnr.update(err.calc_psnr(preds.squeeze().detach().to('cpu').numpy(), labels.squeeze().detach().to('cpu').numpy()), len(images))

    print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

    if epoch_psnr.avg > best_psnr:
        best_epoch = epoch
        best_psnr = epoch_psnr.avg
        best_weights = copy.deepcopy(model.state_dict())


print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
torch.save(best_weights, os.path.join(path))

print('************ Training completed ******************')
end = time.time()
print('total time taken for training is', (end-start))