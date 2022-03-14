from turtle import forward
import torch
import torch.nn as nn
import torchvision
import cv2
import torch.nn.init as init


class ResBlock(nn.Module):

    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        out = x + self.layers(x)
        return out

class CoarseSRNetwork(nn.Module):

    def __init__(self):
        super(CoarseSRNetwork, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
#         self.res_blocks = nn.Sequential(*([ResBlock(64)] * 1))
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Tanh(),
        )
        self.conv3=nn.Sequential(
            nn.ConvTranspose2d(16, 1, 2, stride=2)
        )


    def forward(self, x):
        out = self.conv1(x)
#         out = self.res_blocks(out)  #removed to reduce the size of network
        out = self.conv2(out)
        out = self.conv3(out)
        return out


class HourGlassBlock(nn.Module):
    """
    from cydiachen's implementation
    (https://github.com/cydiachen/FSRNET_pytorch)
    """
    def __init__(self, dim, n):
        super(HourGlassBlock, self).__init__()
        self._dim = dim
        self._n = n
        self._init_layers(self._dim, self._n)

    def _init_layers(self, dim, n):
        setattr(self, 'res'+str(n)+'_1', Residual(dim, dim))
        setattr(self, 'pool'+str(n)+'_1', nn.MaxPool2d(2,2))
        setattr(self, 'res'+str(n)+'_2', Residual(dim, dim))
        if n > 1:
            self._init_layers(dim, n-1)
        else:
            self.res_center = Residual(dim, dim)
        setattr(self,'res'+str(n)+'_3', Residual(dim, dim))
        setattr(self,'unsample'+str(n), nn.Upsample(scale_factor=2))

    def _forward(self, x, dim, n):
        up1 = x
        up1 = eval('self.res'+str(n)+'_1')(up1)
        low1 = eval('self.pool'+str(n)+'_1')(x)
        low1 = eval('self.res'+str(n)+'_2')(low1)
        if n > 1:
            low2 = self._forward(low1, dim, n-1)
        else:
            low2 = self.res_center(low1)
        low3 = low2
        low3 = eval('self.'+'res'+str(n)+'_3')(low3)
        up2 = eval('self.'+'unsample'+str(n)).forward(low3)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._forward(x, self._dim, self._n)



class Residual(nn.Module):
    """
    from cydiachen's implementation
    (https://github.com/cydiachen/FSRNET_pytorch)
    """
    def __init__(self, ins, outs):
        super(Residual, self).__init__()
        hdim = int(outs/2)
        self.convBlock = nn.Sequential(
            nn.BatchNorm2d(ins),
            nn.ReLU(True),
            nn.Conv2d(ins, hdim, 1),
            nn.BatchNorm2d(hdim),
            nn.ReLU(True),
            nn.Conv2d(hdim, hdim, 3, 1, 1),
            nn.BatchNorm2d(hdim),
            nn.ReLU(True),
            nn.Conv2d(hdim, outs, 1)
        )
        if ins != outs:
            self.skipConv = nn.Conv2d(ins, outs, 1)
        self.ins = ins
        self.outs = outs

    def forward(self, x):
        residual = x
        x = self.convBlock(x)
        if self.ins != self.outs:
            residual = self.skipConv(residual)
        x += residual
        return x


class FineSREncoder(nn.Module):

    def __init__(self):
        super(FineSREncoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
#         self.res_blocks = nn.Sequential(*([ResBlock(64)] * 3))
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Tanh(),
        )
        self.conv3=nn.Sequential(
            nn.ConvTranspose2d(32, 64, 2, stride=2)
        )

    def forward(self, x):
        out = self.conv1(x)
#         out = self.res_blocks(out) #removed to reduce size of network
        out = self.conv2(out)
        out = self.conv3(out)
        return out


class PriorEstimationNetwork(nn.Module):

    def __init__(self):
        super(PriorEstimationNetwork, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.res_blocks = nn.Sequential(
            Residual(64, 128),
            ResBlock(128),
        )
        self.hg_blocks = nn.Sequential(
            HourGlassBlock(128, 1),
            nn.Conv2d(128, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(True),    
        )
        self.conv2=nn.Sequential(
            nn.ConvTranspose2d(3, 1, 2, stride=2)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.res_blocks(out)
        out = self.hg_blocks(out)
        out = self.conv2(out)
        return out


class FineSRDecoder(nn.Module):

    def __init__(self):
        super(FineSRDecoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(65, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
#         self.res_blocks = nn.Sequential(*([ResBlock(64)] * 1))
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.deconv1(out) #it doubles the size of input
#         out = self.res_blocks(out)  #removed to reduce size of model
        out = self.conv2(out)
        return out


class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg16_model = torchvision.models.vgg16(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg16_model.features.children())[:15])

    def forward(self, x):
        return self.feature_extractor(x)
    
    
class MRINet(nn.Module):

    def __init__(self,device='cpu'):
        super(MRINet, self).__init__()
        self.csr_net = CoarseSRNetwork().to(device)
        self.fsr_enc = FineSREncoder().to(device)
        self.pre_net = PriorEstimationNetwork().to(device)
        self.fsr_dec = FineSRDecoder().to(device)

        self.concat = None
        self.y_c = None
        self.f = None
        self.p= None

    def forward(self, x):
        y_c = self.csr_net(x)
        f = self.fsr_enc(y_c) ##output of encoder net
        p = self.pre_net(y_c) ## output of prior network
        
        concat = torch.cat((f, p), 1)
        out = self.fsr_dec(concat)

        self.concat = concat
        self.y_c = y_c
        self.f = f
        self.p = p
        return out,f,p,y_c
        
     
class SimpleNetwork(nn.Module):

    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.res_blocks = nn.Sequential(*([ResBlock(64)] * 1))
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Tanh(),
        )
        self.conv3=nn.Sequential(
            nn.ConvTranspose2d(16, 1, 2, stride=2)
        )


    def forward(self, x):
        out = self.conv1(x)
        out = self.res_blocks(out)  #removed to reduce the size of network
        out = self.conv2(out)
        out = self.conv3(out)
        return out

class CoarseNetwork(nn.Module):
    def __init__(self):
        super(CoarseNetwork, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.res_blocks = nn.Sequential(*([ResBlock(64)] * 1))  #residual blocks
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Tanh(),
        )
        self.conv3=nn.Sequential(
            nn.ConvTranspose2d(16, 1, 2, stride=2)
        )
    def forward(self, x):
        out = self.conv1(x)
        out = self.res_blocks(out)  #residual block
        out = self.conv2(out)
        out = self.conv3(out)
        return out

class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2),
            nn.ReLU(True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class SRshuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(SRshuffle, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)


'''in this network we pass both input
     and label from prior estimation network and measure the mse between output 
     for both which will be the parsing error'''
class MRINetV2(nn.Module):
    def __init__(self,device='cpu',mode = 'eval'):
        super(MRINetV2, self).__init__()
        self.csr_net = CoarseSRNetwork().to(device)
        self.fsr_enc = FineSREncoder().to(device)
        self.pre_net = PriorEstimationNetwork().to(device)
        self.fsr_dec = FineSRDecoder().to(device)
        self.mode = mode
        self.concat = None
        self.y_c = None
        self.f = None
        self.p= None
        self.y_p = None

    def forward(self, x , y = None):
        y_c = self.csr_net(x)
        f = self.fsr_enc(y_c) ##output of encoder net
        p = self.pre_net(y_c) ## output of prior network for input

        if self.mode == 'train':
            y_p = self.pre_net(y)
            self.y_p= y_p
        
        concat = torch.cat((f, p), 1)
        out = self.fsr_dec(concat)

        self.concat = concat
        self.y_c = y_c
        self.f = f
        self.p = p
        if self.mode == 'train':
            return out,f,p,y_c,y_p
        else:
            return out


class SRNet(nn.Module):
    def __init__(self):
        super(SRNet,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=8,stride=4,padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,192,kernel_size=2,padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,256,kernel_size=2,padding=2),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256,192,kernel_size=2,padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192,64,kernel_size=2,padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,1,kernel_size=8,stride=4,padding=2),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        x = self.features(x)
        x= self.upsample(x)
        return x

