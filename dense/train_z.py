
import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
# from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import pickle

from densenet import SRDenseNet
from dataset import MRIDataset, RdnSampler
from utils import AverageMeter, calc_psnr,create_dictionary

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value



def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    lr = args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    # log to TensorBoard
    if args.tensorboard:
        log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs-dir', type=str, required=False,default='outputs')
    parser.add_argument('--weights-file', type=str)
    parser.add_argument('--growth-rate', type=int, default=4)
    parser.add_argument('--num-blocks', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--scale', type=int, default=4)
    # parser.add_argument('--patch-size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-epochs', type=int, default=500)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--factor', type=int, default=4)
    parser.add_argument('--name', default='DenseNet_2', type=str,
                    help='name of experiment')
    parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'z-axis')

    if args.tensorboard: 
        run_path = os.path.join('runs', '{}'.format(args.name))
        configure(run_path)
        if not os.path.exists(run_path):
            os.makedirs(run_path)

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    model = SRDenseNet(num_channels=1, growth_rate=args.growth_rate, num_blocks = args.num_blocks, num_layers=args.num_layers).to(device)
    # print(model)
    model = nn.DataParallel(model,device_ids=[0,1,2,3])

    if args.weights_file is not None:
        state_dict = model.state_dict()
        for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
 
    train_batch_size = args.batch_size
    val_batch_size = 1

    print('training for factor ',args.factor)

   
    str_train_2 = 'factor_2/train'
    str_train_4 = 'factor_4/train'
    str_val_2 = 'factor_2/val'
    str_val_4 = 'factor_4/val'

    if args.factor == 2:  
        train_img_dir = 'dataset/z-axis/'+ str(str_train_2)
        val_img_dir = 'dataset/z-axis/'+str(str_val_2)
    elif args.factor == 4:
        train_img_dir = 'dataset/z-axis/'+str(str_train_4)
        val_img_dir = 'dataset/z-axis/'+ str(str_val_4)
    else:
        print('please set the value for factor in [2,4,8]')

    train_label_dir = 'dataset/z-axis/label/train'
    val_label_dir = 'dataset/z-axis//label/val'
    dir_traindict = create_dictionary(train_img_dir,train_label_dir)
    val_dir_traindict = create_dictionary(val_img_dir,val_label_dir)

    train_datasets = MRIDataset(train_img_dir, train_label_dir, dir_dict = dir_traindict,test=False)
    val_datasets = MRIDataset(val_img_dir, val_label_dir, dir_dict = val_dir_traindict,test=True)

    sampler = RdnSampler(train_datasets,train_batch_size,True,classes=train_datasets.classes())
    val_sampler = RdnSampler(val_datasets,val_batch_size,True,classes=train_datasets.classes())

    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size = train_batch_size,sampler = sampler,shuffle=False,
        num_workers=1,pin_memory=False,drop_last=False)
    eval_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size = val_batch_size,sampler = val_sampler,shuffle=False,
        num_workers=1,pin_memory=False,drop_last=False)

    # best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    k=0
    losses =[]
    for epoch in range(args.num_epochs):
        adjust_learning_rate(optimizer, epoch)
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_datasets) - len(train_datasets) % args.batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for idx, (images, labels,parsers) in enumerate(train_dataloader):
                images = images.to(device)
                labels = labels.to(device)
                parsers = parsers.to(device)

                preds = model(images)

                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(images))
                losses.append(loss)
                if args.tensorboard:
                    log_value('train_loss', epoch_losses.avg, epoch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(images))
        if k == 100:
            torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'z_axis_epoch_{}_f_{}.pth'.format(epoch,args.factor)))
            k=0
        k+=1

        model.eval()
        epoch_psnr = AverageMeter()

        for idx, (images, labels,parsers) in enumerate(eval_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                preds = model(images).clamp(0.0, 1.0)

        epoch_psnr.update(calc_psnr(preds.squeeze().detach().to('cpu').numpy(), labels.squeeze().detach().to('cpu').numpy()), len(images))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    
    path = f'dense_zaxis_e'+str(args.num_epochs)+'_f'+str(args.factor)+'_final.pth'

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    # torch.save(best_weights, os.path.join(args.outputs_dir, path))
    torch.save(best_weights, 'outputs/x4/'+str(path))
    print('model saved')
    # torch.save(model.state_dict(), path)
 
    with open("losses", "wb") as fp:   #Pickling
        pickle.dump(losses, fp)