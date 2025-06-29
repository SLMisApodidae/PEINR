import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from torch.optim import LBFGS, Adam
from tqdm import tqdm
import scipy.io
from dataio import *
from models import *
from util import *
from train import *
import argparse
from hash_encoding import HashEncoder
parser = argparse.ArgumentParser(description='PyTorch Implementation of SSR-TVD')
parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                    help='learning rate')
parser.add_argument('--batch_size', type=int, default=20000, metavar='N',
                    help='input batch size for training')
parser.add_argument('--mode', type=str, default='inf' ,
                    help='training or inference')
parser.add_argument('--epochs', type=int, default=4000, metavar='N',
                    help='number of epochs to train (default: 400)')
parser.add_argument('--dim', type=list, default=[120,480], metavar='N',
                    help='low1dim')
parser.add_argument('--dimHigh', type=list, default=[480,1920], metavar='N',
                    help='dimHigh1')
parser.add_argument('--data_pathHigh', type=str, default='weno3/RT1920/', metavar='N',
                    help='data_pathHigh')
parser.add_argument('--data_pathLow', type=str, default='weno3/RT480/', metavar='N',
                    help='data_pathLow')
args = parser.parse_args()

def main(args):
	seed = 0
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	
	if args.mode == 'train':
		dtrain = RT3_16X(args,460, 481,train=True)
		dtest = RT3_16X(args,500, 501)
		dataloader_train = torch.utils.data.DataLoader(dtrain, batch_size=args.batch_size, shuffle=True, num_workers=1)
		dataloader_test = torch.utils.data.DataLoader(dtest, batch_size=args.batch_size, shuffle=False, num_workers=1)
		bounding_box=torch.tensor(([0,0,0.36],[0.25,1,0.96])).to('cuda')
		hash_encoder = HashEncoder(bounding_box)
		hash_encoder.to('cuda')
		model = PEINR(hash_encoder,d_out=1, d_hidden=256, d_model=32, N=1, heads=2).to('cuda')
		trainNet(model,dataloader_train, args,scheduler = nifll_scheduler, validation_data=dataloader_test, save_model_freq=1000)

	elif args.mode == 'inf':
		for i in range(481,500):
			dtest = RT3_16X(args,i, i+1)
			dataloader_test = torch.utils.data.DataLoader(dtest, batch_size=args.batch_size, shuffle=False, num_workers=1)
			bounding_box=torch.tensor(([0,0,0.36],[0.25,1,0.96])).to('cuda')
			hash_encoder = HashEncoder(bounding_box)
			hash_encoder.to('cuda')
			model = PEINR(hash_encoder,d_out=1, d_hidden=256, d_model=32, N=1, heads=2).to('cuda')
			infNet(model,args,i,validation_data=dataloader_test)
	
if __name__== "__main__":
    main(args)





