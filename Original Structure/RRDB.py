# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 20:40:26 2020

@author: Mateo
"""

import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import cv2
import math
from piq import MultiScaleSSIMLoss
from piq import psnr, ssim
from PIL import Image
import gzip
import bz2

numc = 32
qdtype = torch.qint32
dec = 0.01
maxminto = 0
op = 'AdamW'

plt.rcParams['axes.grid'] = False
plt.rcParams['figure.dpi'] = 1200

#torch.set_num_threads(3)
torch.set_printoptions(4) #35
torch.backends.cudnn.benchmark = True 
ss = 168
#stats = ((0.2239, 0.2144, 0.2166), (0.2933, 0.2852, 0.2889))
train_tfms = tt.Compose([#tt.RandomCrop(32, padding=4, padding_mode='reflect'), 
                        #tt.RandomVerticalFlip(), 
                        tt.Resize((ss,248)),
                        tt.ToTensor()])
                        #tt.Normalize(*stats,inplace=True)])
                        
valid_tfms = tt.Compose([tt.Resize((ss,248)), tt.ToTensor() ])
#train_ds, val_ds = torch.utils.data.random_split(dataset, [15184, 1001])   #16185 -> 15184 1001 -> 16*13*73 7*11*13 -> batch 16, batch 77
train_ds = ImageFolder('/content/drive/My Drive/cars_tv/train', train_tfms)
val_ds = ImageFolder('/content/drive/My Drive/cars_tv/valid', train_tfms)
valid_ds = ImageFolder('/content/drive/My Drive/tiff', valid_tfms)

print('DATASET: '+str(len(train_ds)) +" "+ str(len(val_ds)))
print('CONFIGURATION: ' + str(numc) +" "+ str(qdtype) +" "+ str(dec) +" "+ 'Maxminavgto' +" "+ str(maxminto) +" "+ op)

#15 or 13 or 195
batch_size = 20 
train_dl = DataLoader(train_ds, batch_size, pin_memory = True, shuffle=True, num_workers=2)
val_dl = DataLoader(val_ds, 20, pin_memory = True, num_workers=2)
valid_dl = DataLoader(valid_ds, 4)

class STEFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        lquant = torch.quantize_per_tensor(input, dec, 0, dtype=qdtype) 
        unqlat = torch.dequantize(lquant)
        return unqlat

    @staticmethod
    def backward(ctx, grad_output):
        #return F.hardtanh(grad_output)
        return grad_output

class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)
        #self.RDB4 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        #out = self.RDB4(out)
        return out * 0.2 + x

class ResidualDenseBlock_5C_dec(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C_dec, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.ConvTranspose2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.ConvTranspose2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.ConvTranspose2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.ConvTranspose2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.ConvTranspose2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB_dec(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB_dec, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C_dec(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C_dec(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C_dec(nf, gc)
        #self.RDB4 = ResidualDenseBlock_5C_dec(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        #out = self.RDB4(out)
        return out * 0.2 + x


class resConv(nn.Module):
    def __init__(self):
        super(resConv, self).__init__()
        
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 5, stride=2, padding=2,padding_mode='reflect'),
                                  nn.LeakyReLU()
                                  )
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 5, stride=2, padding=2,padding_mode='reflect')
                                  )
        self.RinR = RRDB(nf=128, gc=256)

        #latent
        self.enc6 = nn.Conv2d(128, numc, 5, stride=2, padding=2,padding_mode='reflect')

        self.lat = StraightThroughEstimator()
        
        #DEC
        self.px1 = nn.Sequential(nn.ConvTranspose2d(numc, 512, 3, stride=1, padding=1), 
                                 nn.PixelShuffle(2)
                                 )
        
        self.RinRdec = RRDB_dec(nf=128, gc=256)

        self.px2 = nn.Sequential(nn.ConvTranspose2d(128, 256, 3, stride=1, padding=1), 
                                 nn.PixelShuffle(2),
                                 nn.LeakyReLU()
                                 )
        
        
        self.px3 = nn.Sequential(nn.ConvTranspose2d(64, 12, 3, stride=1, padding=1),
                                 nn.PixelShuffle(2),
                                 )
        
        
    def encoder(self,x):
        oute = self.enc1(x)     
        oute = self.enc2(oute)
        oute = self.RinR(oute)
        latent = self.enc6(oute)
        unqlat = self.lat(latent)
        return latent,unqlat
        
    def decoder(self,unqlat):
        outd = self.px1(unqlat)
        outd = self.RinRdec(outd)
        outd = self.px2(outd)
        out = self.px3(outd)
        out = out.clamp(0,1)
        return out
    
    def forward(self,x):
        latent,unqlat = self.encoder(x)
        out = self.decoder(unqlat)
        return out, latent, unqlat

device = 'cuda'
autoenc = resConv()
autoenc.load_state_dict(torch.load('/content/drive/My Drive/epochs/autoenc00177.pth'))
autoenc.to(device)
criterion = MultiScaleSSIMLoss()  
vcrit = MultiScaleSSIMLoss()
#Epochs
n_epochs = 10000
init_lr = 0.0002
optimizer = torch.optim.AdamW(autoenc.parameters(), lr=init_lr)
clipping_value = 1 # arbitrary value of your choosing
lr = init_lr

def final_loss(mssim_loss, latent):
    MSSIM = mssim_loss 
    Qdifmax = (0 - latent.max())**2
    Qdifmin = (0 - latent.min())**2
    return MSSIM + Qdifmax + Qdifmin

for epoch in range(178, n_epochs+1):
    train_loss = 0.0
    val_loss= 0.0
    print(epoch)
    
    ps,ss = [], []
    autoenc.train()
    for data in train_dl:
        images, _ = data
        images = images.to(device)
        outputs, latent, a = autoenc(images)
        loss = criterion(outputs, images)
        loss = final_loss(loss, latent)
        optimizer.zero_grad()
        loss.backward()        
        torch.nn.utils.clip_grad_norm_(autoenc.parameters(), clipping_value)
        optimizer.step()
        train_loss += loss.item()*images.size(0)
        ps.append(psnr(outputs,images).to('cpu').detach().numpy())
        ss.append(ssim(outputs,images).to('cpu').detach().numpy())
      
    train_loss = train_loss/len(train_dl)
    print('Epoch: {} T Loss: {:.6f}'.format(epoch, train_loss) + " %" + str(np.exp(-abs(train_loss))*100) + " " + str(100-train_loss*100))
    print(np.mean(np.array(ps)), np.mean(np.array(ss)))
    torch.save(autoenc.state_dict(), 'autoenc{0:05d}.pth'.format(epoch))
    torch.save(autoenc.state_dict(), '/content/drive/My Drive/epochs/autoenc{0:05d}.pth'.format(epoch))

    ps,ss = [], []
    autoenc.eval()
    with torch.no_grad():
        for datav in val_dl:
            imagesv,_ = datav
            imagesv = imagesv.to(device)
            outputsv, latentv, a = autoenc(imagesv)
            lossv = vcrit(outputsv, imagesv)
            lossv = final_loss(lossv, latentv)
            val_loss += lossv.item()*imagesv.size(0)
            ps.append(psnr(outputsv,imagesv).to('cpu').detach().numpy())
            ss.append(ssim(outputsv,imagesv).to('cpu').detach().numpy())

        val_loss = val_loss/len(val_dl)
        print('Epoch: {} V Loss: {:.6f}'.format(epoch, val_loss) + " %" + str(np.exp(-abs(val_loss))*100) + " " + str(100-val_loss*100))
        print(np.mean(np.array(ps)), np.mean(np.array(ss)))
    
        if epoch%5 ==0:
            #Batch of test images
            dataiter = iter(valid_dl)
            images, labels = dataiter.next()
            images = images.to(device)
            #validation
            batch_size=1
            outputV, aaaaa, latentcopy = autoenc(images)
            #encoder output
            images = images.to('cpu')
            outputV = outputV.to('cpu')
            print(psnr(outputV[0].unsqueeze(0), images[0].unsqueeze(0)),ssim(outputV[0].unsqueeze(0),images[0].unsqueeze(0)))
            print(psnr(outputV[1].unsqueeze(0), images[1].unsqueeze(0)),ssim(outputV[1].unsqueeze(0),images[1].unsqueeze(0)))
            print(psnr(outputV[2].unsqueeze(0), images[2].unsqueeze(0)),ssim(outputV[2].unsqueeze(0),images[2].unsqueeze(0)))
            print(psnr(outputV[3].unsqueeze(0), images[3].unsqueeze(0)),ssim(outputV[3].unsqueeze(0),images[3].unsqueeze(0)))

#Batch of test images
dataiter = iter(valid_dl)
images, labels = dataiter.next()
images = images.to(device)
#validation
batch_size=1
outputV, aaaaa, latentcopy = autoenc(images)
#encoder output

images = images.to('cpu')
outputV = outputV.to('cpu')

outputVnp = outputV.detach().numpy()
outputVnp = np.transpose(outputVnp, (0,2,3,1))
#real images
imagesnp = images.detach().numpy()
imagesnp = np.transpose(imagesnp, (0,2,3,1))

plt.figure()
f, plot = plt.subplots(1,2)
plot[0].imshow(outputVnp[0])
plot[1].imshow(imagesnp[0])
plt.figure()
f, plot = plt.subplots(1,2)
plot[0].imshow(outputVnp[1])
plot[1].imshow(imagesnp[1])
plt.figure()
f, plot = plt.subplots(1,2)
plot[0].imshow(outputVnp[2])
plot[1].imshow(imagesnp[2])
plt.figure()
f, plot = plt.subplots(1,2)
plot[0].imshow(outputVnp[3])
plot[1].imshow(imagesnp[3])

print(psnr(outputV[0].unsqueeze(0), images[0].unsqueeze(0)),ssim(outputV[0].unsqueeze(0),images[0].unsqueeze(0)))
print(psnr(outputV[1].unsqueeze(0), images[1].unsqueeze(0)),ssim(outputV[1].unsqueeze(0),images[1].unsqueeze(0)))
print(psnr(outputV[2].unsqueeze(0), images[2].unsqueeze(0)),ssim(outputV[2].unsqueeze(0),images[2].unsqueeze(0)))
print(psnr(outputV[3].unsqueeze(0), images[3].unsqueeze(0)),ssim(outputV[3].unsqueeze(0),images[3].unsqueeze(0)))

outsave = outputVnp*255
outsave = outsave.astype('uint8')
print(outsave.shape, outsave.dtype)
latentcopy = latentcopy.to('cpu')
latnp = latentcopy.detach().numpy()

latnq = aaaaa.to('cpu')
latnpnq = latnq.detach().numpy()

for i in range(0, 4):
    im2 = Image.fromarray(outsave[i])
    im2.save('/content/drive/My Drive/caeouttiff{0:05d}.png'.format(i))
    #latnp = latentcopy
    latnps = latnp[i].squeeze()
    print(latnps.shape)
    #np.save('/content/drive/My Drive/DATA/lat_np{0:05d}'.format(i), latnps, allow_pickle=True)
    np.save('/content/drive/My Drive/DATA/lat_np{0:05d}'.format(i), latnps, allow_pickle=True)

    f = gzip.GzipFile('/content/drive/My Drive/DATA/lat_gz{0:05d}.gz'.format(i), "w", compresslevel=9)
    np.save(file=f, arr=latnps)
    f.close()

    f = bz2.BZ2File('/content/drive/My Drive/DATA/lat_bz2{0:05d}.bz2'.format(i), "w", compresslevel=9)
    np.save(file=f, arr=latnps)
    f.close()

    latnpsq = latnpnq[i].squeeze()
    np.save('/content/drive/My Drive/DATA/lat_notq{0:05d}'.format(i), latnpsq, allow_pickle=True)