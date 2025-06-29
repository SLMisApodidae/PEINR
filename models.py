import torch
import torch.nn as nn
import pdb
import time
import math
from util import get_clones
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import init
from FlashMHA import *
from wire import *
import torch.nn.utils.spectral_norm as spectral_norm
from spectformer import *
class Sine(nn.Module):
    def __init(self):
        super(Sine,self).__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()

    
    def init_weights(self):
        with torch.no_grad():
            self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features) 
            #self.linear.weight.normal_(0,0.05) 
        
    def forward(self, input):
        return self.linear(input)

class ResBlock(nn.Module):
    def __init__(self,in_features,out_features,nonlinearity='relu'):
        super(ResBlock,self).__init__()
        nls_and_inits = {'sine':Sine(),
                         'relu':nn.ReLU(inplace=True),
                         'sigmoid':nn.Sigmoid(),
                         'tanh':nn.Tanh(),
                         'selu':nn.SELU(inplace=True),
                         'softplus':nn.Softplus(),
                         'elu':nn.ELU(inplace=True)}

        self.nl = nls_and_inits[nonlinearity]

        self.net = []

        self.net.append(SineLayer(in_features,out_features))

        self.net.append(SineLayer(out_features,out_features))

        self.flag = (in_features!=out_features)

        if self.flag:
            self.transform = SineLayer(in_features,out_features)

        self.net = nn.Sequential(*self.net)
    
    def forward(self,features):
        outputs = self.net(features)
        if self.flag:
            features = self.transform(features)
        return 0.5*(outputs+features)
class PEINR(nn.Module):
	def __init__(self,hash_encoder, d_out, d_model, d_hidden, N, heads):
		super(PEINR, self).__init__()
		self.num_res = 5
		init_features = 64
		self.net3 = []
		self.net3.append(ResBlock(5,2*init_features))
		self.net3.append(ResBlock(2*init_features,2*init_features))
		self.net3.append(ResBlock(2*init_features,4*init_features))

	
		for i in range(self.num_res):
			self.net3.append(ResBlock(4*init_features,4*init_features))
		self.net3 = nn.Sequential(*self.net3)
		
		self.attn  = nn.MultiheadAttention(embed_dim=4*init_features, num_heads=heads, batch_first=True)
		self.attn1 = nn.MultiheadAttention(embed_dim=4*init_features, num_heads=heads, batch_first=True)
		self.attn2 = nn.MultiheadAttention(embed_dim=4*init_features, num_heads=heads, batch_first=True)

		self.SpectG1=PreNorm(4*init_features, SpectralGatingNetwork(4*init_features, h = 2, w = 2))
		self.FF1    =PreNorm(4*init_features, FeedForward(4*init_features, 4*init_features, dropout = 0., drop_path = 0.))

		self.SpectG2=PreNorm(4*init_features, SpectralGatingNetwork(4*init_features, h = 2, w = 2))
		self.FF2    =PreNorm(4*init_features, FeedForward(4*init_features, 4*init_features, dropout = 0., drop_path = 0.))
				
		self.net4 = []
		for i in range(5):
			self.net4.append(ResBlock(4*init_features,4*init_features))
		self.net4 = nn.Sequential(*self.net4)
		
		self.net5 = []
		for i in range(5):
			self.net5.append(ResBlock(4*init_features,4*init_features))
		self.net5 = nn.Sequential(*self.net5)
		
		self.net = []
		self.net.append(ResBlock(4*init_features,d_out))
		self.net = nn.Sequential(*self.net)

	def forward(self, x, y,t1,t2,t3):
		xt = torch.cat((t1,t2,t3), dim=-1)
		yt = torch.cat((x,y), dim=-1)

		xyt = torch.cat((xt,yt), dim=-1)
		output3 = self.net3(xyt)
		# output3 = self.SpectG1(output3)
		# output3 = self.FF1(output3)
		o1 = output3+self.attn(output3,output3,output3)[0]
		o3 = self.net4(o1)
		# o2 = self.SpectG2(o2)
		# o3 = self.FF2(o2)
		o4 = o3+self.attn1(o3,o3,o3)[0]
		o4 = self.net5(o4)
		o5 = o4+self.attn2(o4,o4,o4)[0]
		o6 = self.net(o5)
		return o6
