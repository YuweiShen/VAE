# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 19:55:54 2019

@author: susie
"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as fun
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# cuda

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# hyperparameters

BATCH = 64
lr = 1e-3
LATENT_DIM = 2
encoder_dim = 32

decoder_dim = 64
input_dim = 32


## 加载数据
train_data = pd.read_csv('C:/Users/58454/Desktop/GAN/data/nltcs_train.csv')
test_data = pd.read_csv('C:/Users/58454/Desktop/GAN/data/nltcs_test.csv')
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH, shuffle=True)
#dtest = dtest.view([7192,1,16])


class VAE(nn.Module):
    def __ini__(self,data_size = 1, encoder_dim = 128, decoder_dim = 64):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
                nn.Conv1d(in_channels = data_size,out_channels = input_dim,kernel_size = 3 ),
                nn.BatchNorm1d(BATCH),
                nn.ReLU(),
                
                nn.Conv1d(input_dim, encoder_dim, 3),
                nn.BatchNorm1d(BATCH),
                nn.ReLU()
                )
        self.fcmu = nn.Linear(encoder_dim * 7 * 7, LATENT_DIM) #均值
        self.fcsigma = nn.Linear(encoder_dim * 7 * 7, LATENT_DIM) #标准差
        self.fc = nn.Linear(LATENT_DIM, encoder_dim1 * 7 * 7)
        self.decoder = nn.Sequential(              
                nn.ConvTranspose1d(encoder_dim, decoder_dim, kernel_size=3),
                nn.ReLU(),
                nn.ConvTranspose1d(decoder_dim, data_size , kernel_size=3),
                nn.Sigmoid()
                )

    def reparameterize(self,mu,logvar):
        epsi = Variable(torch.random(mu.size(0), mu.size(1))).cuda()
        z = mu + epsi*torch.exp(logvar/2)
        return z
    
    def forward(self,x):
        out1, out2 =self.encoder(x), self.encoder(x)
        mu = self.fcmu(out1.view(out1.size(0),-1))
        logvar = self.fcsigma(out2.view(out2.size(0),-1))
        z = self.reparameterize(mu, logvar)
        out3 = self.fc(z).view(z.size(0), encoder_dim, 7, 7)
        return(self.decoder(out3), mu, logvar)

def loss_func(reconstruct, x, mu, logvar):
     BCE = fun.binary_cross_entropy(reconstruct, x,  size_average=False)
     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
     return BCE+KLD
 
vae = VAE().cuda()

def vaetrain(t, path, vae, dataloader, lr, epochs, steps_per_epoch, GPU=True):
    optimizer =  optim.Adam(vae.parameters(), lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00001)
    for epoch in range(epochs):
        print("----------pretrain Epoch %d:----------\n"%epoch)
		log = open(path+"train_log_"+str(t)+".txt","a+")
		log.write("----------pretrain Epoch %d:----------\n"%epoch)
		log.close()
		it = 0
		while it < steps_per_epoch:
			for x in dataloader:
				if GPU:
					x = Variable(x).cuda()
				else:
					x = Variable(x)
                optimizer.zero_grad()
                x_, mu, logvar = vae.forward(x)
                loss = loss_func(x_, x, mu, logvar)
                loss.backward()
                optimizer.step()
                
            if it%100 == 0:
                print("VAE iteration {}, loss: {}\n".format(it, loss.data))
                log = open(path+"train_log_"+str(t)+".txt","a+")
                log.write("VAE iteration {}, loss: {}\n".format(it, loss.data))
				log.close()
            it += 1
            if it >= steps_per_epoch:
                break
