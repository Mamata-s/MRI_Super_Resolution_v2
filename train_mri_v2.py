
import torch
import torch.nn as nn
from torchvision import transforms
import  cv2
import model_concat as mconcat
import model_utility as ut

# Load the training dataset
train_batch_size =32
train_img_dir = 'resolution_dataset/train_8/input'
train_label_dir = 'resolution_dataset/train_8/label'
dir_traindict = ut.create_dictionary(train_img_dir,train_label_dir)
trans = transforms.Compose([
   transforms.CenterCrop(400),
])
train_datasets = ut.MRIDataset(train_img_dir, train_label_dir, dir_dict = dir_traindict,test=False)
train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size=train_batch_size, shuffle=True)


loss_fn = nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device available is ',device)


# hyperparameters
batch_size = 32
channels = 1
n_epochs = 700
learning_rate = 0.01

lbda = 0.3
alpha = 0.5
beta = 0.2


model_concat = mconcat.MRINetV2(device,mode='train')
model_concat = nn.DataParallel(model_concat,device_ids=[0,1,2,3])
model_concat.to(device)
# print(model_concat)

optim = torch.optim.SGD(params = model_concat.parameters(), lr = learning_rate, momentum=0.9)

model_concat.train() 
losses_concat = []
k = 0
print('training started')
for epoch in range(1, n_epochs + 1):
    for idx, (images, labels,parsers) in enumerate(train_dataloaders):
        images = images.to(device)
        labels = labels.to(device)
        parsers = parsers.to(device)
        torch.cuda.empty_cache()

        outputs,f,p,y_c, y_p = model_concat(images,labels)
        loss_coarser = loss_fn(y_c, labels) 
        loss_decoder = loss_fn(outputs, labels) 
        loss_parser = loss_fn(p,y_p) 
        loss = (loss_coarser*beta) + (alpha * loss_decoder) + (loss_parser * lbda)
        
        optim.zero_grad()
        loss.backward()
        optim.step()

        losses_concat.append(float(loss))

        if idx % 200 == 0:
            print("Epoch [%d/%d]. Iter [%d/%d]. Loss: %0.2f" % (epoch, n_epochs, idx + 1, len(train_dataloaders), loss))

    if k == 500:
        path = f'model/concat/model_v2_concat_'+str(epoch)+'.pth'
        torch.save(model_concat.state_dict(), path)
        k=0
    k+=1

print('Training Completed')
path = f'model/concat/model_v2_concat_'+'_final.pth'
torch.save(model_concat.state_dict(), path)
print('model saved')









