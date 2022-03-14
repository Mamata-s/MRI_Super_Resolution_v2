import os
import time

import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from dataset import MRIDataset, RdnSampler
from model import Discriminator, Generator, ContentLoss
import argparse
from utils import AverageMeter, calc_psnr,create_dictionary
# used for logging to TensorBoard
from tensorboard_logger import configure, log_value
import config


def build_model(device) -> nn.Module:
    """Building discriminator and generators model
    Returns:
        SRGAN model
    """
    discriminator = Discriminator().to(device)
    generator = Generator().to(device)

    return discriminator, generator


def define_loss(device) -> [nn.MSELoss, nn.MSELoss, ContentLoss, nn.BCEWithLogitsLoss]:
    """Defines all loss functions
    Returns:
        PSNR loss, pixel loss, content loss, adversarial loss
    """
    psnr_criterion = nn.MSELoss().to(device)
    pixel_criterion = nn.MSELoss().to(device)
    content_criterion = ContentLoss().to(device)
    adversarial_criterion = nn.BCEWithLogitsLoss().to(device)

    return psnr_criterion, pixel_criterion, content_criterion, adversarial_criterion


def define_optimizer(discriminator: nn.Module, generator: nn.Module) -> [optim.Adam, optim.Adam]:
    """Define all optimizer functions
    Args:
        discriminator (nn.Module): Discriminator model
        generator (nn.Module): Generator model
    Returns:
        SRGAN optimizer
    """
    d_optimizer = optim.Adam(discriminator.parameters(), config.d_model_lr, config.d_model_betas)
    g_optimizer = optim.Adam(generator.parameters(), config.g_model_lr, config.g_model_betas)

    return d_optimizer, g_optimizer


def define_scheduler(d_optimizer: optim.Adam, g_optimizer: optim.Adam) -> [lr_scheduler.StepLR, lr_scheduler.StepLR]:
    """Define learning rate scheduler
    Args:
        d_optimizer (optim.Adam): Discriminator optimizer
        g_optimizer (optim.Adam): Generator optimizer
    Returns:
        SRGAN model scheduler
    """
    d_scheduler = lr_scheduler.StepLR(d_optimizer, config.d_optimizer_step_size, config.d_optimizer_gamma)
    g_scheduler = lr_scheduler.StepLR(g_optimizer, config.g_optimizer_step_size, config.g_optimizer_gamma)

    return d_scheduler, g_scheduler

def train(discriminator,
          generator,
          train_dataloader,
          psnr_criterion,
          pixel_criterion,
          content_criterion,
          adversarial_criterion,
          d_optimizer,
          g_optimizer,
          epoch,
          scaler) -> None:
    # Calculate how many iterations there are under epoch
    batches = len(train_dataloader)

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    pixel_losses = AverageMeter("Pixel loss", ":6.6f")
    content_losses = AverageMeter("Content loss", ":6.6f")
    adversarial_losses = AverageMeter("Adversarial loss", ":6.6f")
    d_hr_probabilities = AverageMeter("D(HR)", ":6.3f")
    d_sr_probabilities = AverageMeter("D(SR)", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    progress = ProgressMeter(batches,
                             [batch_time, data_time,
                              pixel_losses, content_losses, adversarial_losses,
                              d_hr_probabilities, d_sr_probabilities,
                              psnres],
                             prefix=f"Epoch: [{epoch + 1}]")

    # Put all model in train mode.
    discriminator.train()
    generator.train()

    end = time.time()
    for idx, (lr, hr,parsers) in enumerate(eval_dataloader):
        # measure data loading time
        data_time.update(time.time() - end)

        # Send data to designated device
        lr = lr.to(config.device, non_blocking=True)
        hr = hr.to(config.device, non_blocking=True)

        # Set the real sample label to 1, and the false sample label to 0
        real_label = torch.full([lr.size(0), 1], 1.0, dtype=lr.dtype, device=config.device)
        fake_label = torch.full([lr.size(0), 1], 0.0, dtype=lr.dtype, device=config.device)

        # Use generators to create super-resolution images
        sr = generator(lr)

        # Start training discriminator
        # At this stage, the discriminator needs to require a derivative gradient
        for p in discriminator.parameters():
            p.requires_grad = True

        # Initialize the discriminator optimizer gradient
        d_optimizer.zero_grad()

        # Calculate the loss of the discriminator on the high-resolution image
        with amp.autocast():
            hr_output = discriminator(hr)
            # print('discriminator output for hr', hr_output.item())
            d_loss_hr = adversarial_criterion(hr_output, real_label)
        # Gradient zoom
        scaler.scale(d_loss_hr).backward()

        # Calculate the loss of the discriminator on the super-resolution image.
        with amp.autocast():
            sr_output = discriminator(sr.detach())
            # print('discriminator output for sr', sr_output.item())
            d_loss_sr = adversarial_criterion(sr_output, fake_label)
        # Gradient zoom
        scaler.scale(d_loss_sr).backward()
        # Update discriminator parameters
        scaler.step(d_optimizer)
        scaler.update()

        # Count discriminator total loss
        d_loss = d_loss_hr + d_loss_sr
        # print('discriminator adversial loss', d_loss.item())
        # End training discriminator

        # Start training generator
        # At this stage, the discriminator no needs to require a derivative gradient
        for p in discriminator.parameters():
            p.requires_grad = False

        # Initialize the generator optimizer gradient
        g_optimizer.zero_grad()

        # Calculate the loss of the generator on the super-resolution image
        with amp.autocast():
            output = discriminator(sr)
            pixel_loss = config.pixel_weight * pixel_criterion(sr, hr.detach())
            content_loss = config.content_weight * content_criterion(sr, hr.detach())
            adversarial_loss = config.adversarial_weight * adversarial_criterion(output, real_label)
        # Count discriminator total loss
        g_loss = pixel_loss + content_loss + adversarial_loss
        # Gradient zoom
        scaler.scale(g_loss).backward()
        # Update generator parameters
        scaler.step(g_optimizer)
        scaler.update()

        # End training generator

        # Calculate the scores of the two images on the discriminator
        d_hr_probability = torch.sigmoid(torch.mean(hr_output))
        d_sr_probability = torch.sigmoid(torch.mean(sr_output))

        # print('probability for hr image',d_hr_probability)
        # print('probability for sr image', d_sr_probability)
        # measure accuracy and record loss
        psnr = 10. * torch.log10(1. / psnr_criterion(sr, hr))
        pixel_losses.update(pixel_loss.item(), lr.size(0))
        content_losses.update(content_loss.item(), lr.size(0))
        adversarial_losses.update(adversarial_loss.item(), lr.size(0))
        d_hr_probabilities.update(d_hr_probability.item(), lr.size(0))
        d_sr_probabilities.update(d_sr_probability.item(), lr.size(0))
        psnres.update(psnr.item(), lr.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        iters = idx + epoch * batches + 1
        if idx % config.print_frequency == 0 and idx != 0:
            progress.display(idx)


def validate(model, valid_dataloader, psnr_criterion, epoch) -> float:
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    progress = ProgressMeter(len(valid_dataloader), [batch_time, psnres], prefix="Valid: ")

    # Put the generator in verification mode.
    model.eval()

    with torch.no_grad():
        end = time.time()
        for index, (lr, hr,parsers)in enumerate(valid_dataloader):
            lr = lr.to(config.device, non_blocking=True)
            hr = hr.to(config.device, non_blocking=True)

            # Mixed precision
            with amp.autocast():
                sr = model(lr)

            # measure accuracy and record loss
            psnr = 10. * torch.log10(1. / psnr_criterion(sr, hr))
            psnres.update(psnr.item(), hr.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if index % config.print_frequency == 0:
                progress.display(index)

        # Print evaluation indicators.
        print(f"* PSNR: {psnres.avg:4.2f}.\n")

    return psnres.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=False, default='results_srgan')
    parser.add_argument('--sample-dir', type=str, required=False, default='sample_srgan')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-epochs', type=int, default=300)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--factor', type=int, default=4)
    parser.add_argument('--name', default='SRGAN_2', type=str,
                    help='name of experiment')
    parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
    args = parser.parse_args()

    args.results_dir = os.path.join(args.results_dir, 'x{}'.format(args.factor))

    if args.tensorboard: 
        run_path = os.path.join('runs', '{}'.format(args.name))
        configure(run_path)
        if not os.path.exists(run_path):
            os.makedirs(run_path)

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)


    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    print("Load train dataset and valid dataset...")
    if args.factor == 2: 
        print('training for factor 2 resolution') 
        train_img_dir = '../resolution_dataset/dataset/factor_2/train'
        val_img_dir = '../resolution_dataset/dataset/factor_2/val'
    elif args.factor == 4:
        train_img_dir = '../resolution_dataset/dataset/factor_4/train'
        val_img_dir = '../resolution_dataset/dataset/factor_4/val'
    elif args.factor == 8:
        train_img_dir = '../resolution_dataset/dataset/factor_8/train'
        val_img_dir = '../resolution_dataset/dataset/factor_8/val'
    else:
        print('please set the value for factor in [2,4,8]')

    train_batch_size = args.batch_size
    val_batch_size = 1


    train_label_dir = '../resolution_dataset/dataset/label/train'
    val_label_dir = '../resolution_dataset/dataset/label/val'
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
    print("Load train dataset and valid dataset successfully.")


    print("Build SRGAN model...")
    discriminator, generator = build_model(device)
    print("Build SRGAN model successfully.")

    print("Build SRGAN model...")
    discriminator, generator = build_model(device)
    print("Build SRGAN model successfully.")

    print("Define all loss functions...")
    psnr_criterion, pixel_criterion, content_criterion, adversarial_criterion = define_loss(device)
    print("Define all loss functions successfully.")

    print("Define all optimizer functions...")
    d_optimizer, g_optimizer = define_optimizer(discriminator, generator)
    print("Define all optimizer functions successfully.")

    print("Define all optimizer scheduler functions...")
    d_scheduler, g_scheduler = define_optimizer(discriminator, generator)
    print("Define all optimizer scheduler functions successfully.")


    # Initialize the gradient scaler.
    scaler = amp.GradScaler()

    # Initialize training to generate network evaluation indicators
    best_psnr = 0.0

    print("Start train SRGAN model.")
    for epoch in range(config.start_epoch, args.num_epochs):
        train(discriminator,
              generator,
              train_dataloader,
              psnr_criterion,
              pixel_criterion,
              content_criterion,
              adversarial_criterion,
              d_optimizer,
              g_optimizer,
              epoch,
              scaler)

        psnr = validate(generator, eval_dataloader, psnr_criterion, epoch)
        # Automatically save the model with the highest index
        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)
        torch.save(discriminator.state_dict(), os.path.join(args.sample_dir, f"d_epoch_{epoch + 1}.pth"))
        torch.save(generator.state_dict(), os.path.join(args.sample_dir, f"g_epoch_{epoch + 1}.pth"))
        if is_best:
            torch.save(discriminator.state_dict(), os.path.join(args.results_dir, "d-best.pth"))
            torch.save(generator.state_dict(), os.path.join(args.results_dir, f"g-best.pth"))

        # Update LR
        d_scheduler.step()
        g_scheduler.step()

    # Save the generator weight under the last Epoch in this stage
    torch.save(discriminator.state_dict(), os.path.join(args.results_dir, "d-last.pth"))
    torch.save(generator.state_dict(), os.path.join(args.results_dir, "g-last.pth"))