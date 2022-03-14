"""Realize the parameter configuration function of dataset, model, training and verification code."""
import torch
from torch.backends import cudnn as cudnn

# ==============================================================================
# General configuration
# ==============================================================================
# Random seed to maintain reproducible results
torch.manual_seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# Image magnification factor
upscale_factor = 4
# Current configuration parameter method
mode = "train_srresnet"
# Experiment name, easy to save weights and log files
exp_name = "train_SRResNet_baseline"


# ==============================================================================
# Training SRGAN model configuration
# ==============================================================================


image_size = 96
batch_size = 16
num_workers = 4

# Incremental training and migration training
resume = True
strict = False
start_epoch = 0
resume_d_weight = ""
resume_g_weight = "results/Train_SRResNet_baseline/g-last.pth"

# Total num epochs
epochs = 8

# Loss function weight
pixel_weight = 1.0
content_weight = 1.0
adversarial_weight = 0.001

# Adam optimizer parameter for Discriminator
d_model_lr = 1e-4
g_model_lr = 1e-4
d_model_betas = (0.9, 0.999)
g_model_betas = (0.9, 0.999)

# MultiStepLR scheduler parameter for SRGAN
d_optimizer_step_size = epochs // 2
g_optimizer_step_size = epochs // 2
d_optimizer_gamma = 0.1
g_optimizer_gamma = 0.1

# Print the training log every one hundred iterations
print_frequency = 1000
