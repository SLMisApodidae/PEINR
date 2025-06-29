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
import datetime
def saveFile2Duw(outputFilePath,datau,coords,dim):
	print(coords.shape)

	f=open(outputFilePath,"w")
	
	f.write(" variables=\"x\",\"y\",\"u\" \n" )
	f.write("zone i= "+ '{:4d}'.format(dim[0])+" ,j= "+'{:4d}\n'.format(dim[1]))
	index=0
	for j in range(dim[1]):
		for i in range(dim[0]):
			f.write("%.7f %.7f %.7f\n"%(float(coords[index][1]),float(coords[index][0]),float(datau[index])))
			index+=1
	f.close()
def saveFile2Dxyvali(outputFilePath,data,x,y,dim):
	f=open(outputFilePath,"w")
	
	f.write(" variables=\"x\",\"y\",\"u\" \n" )
	f.write("zone i= "+ '{:4d}'.format(dim[1])+" ,j= "+'{:4d}\n'.format(dim[0]))
	index=0
	for j in range(dim[1]):
		for i in range(dim[0]):
			f.write("%.7f %.7f %.7f\n"%(float(y[index]),float(x[index]),float(data[index])))
			index+=1
		
	f.close()

def get_mgrid(sidelen, dim=2, s=1,t=0):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)
    if dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0]:(t+1),0:1:sidelen[2]*1j, 0:0.25:sidelen[1]*1j], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
    elif dim == 2:
        pixel_coords = np.stack(np.mgrid[0:1:sidelen[1]*1j, 0:0.25:sidelen[0]*1j], axis=-1)[None, ...].astype(np.float32)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords = np.reshape(pixel_coords,(-1,dim))
    return pixel_coords

def validate(args,model,criterion, dataloader,saveResult):
	epoch_loss = 0
	u = []
	x = []
	y = []


	for idx, batch in enumerate(dataloader):
		batch_t1,batch_t2,batch_t3, batch_x, batch_y, batch_data= batch
		batch_t1 = batch_t1.cuda()
		batch_t2 = batch_t2.cuda()
		batch_t3 = batch_t3.cuda()
		batch_x = batch_x.cuda()
		batch_y = batch_y.cuda()
		
		batch_data = batch_data.cuda()

		vhat = model(batch_x,batch_y,batch_t1,batch_t2,batch_t3)

		vhat = vhat[:,0,0]
		vT = batch_data[:,0,0]
		xhat = batch_x[:,0,0]
		yhat = batch_y[:,0,0]

		loss = criterion(vhat, vT)
		u += list(vhat.view(-1).detach().cpu().numpy())
		x += list(xhat.view(-1).detach().cpu().numpy())	
		y += list(yhat.view(-1).detach().cpu().numpy())

		epoch_loss += loss.mean().item()

		
	u = np.asarray(u,dtype='<f')
	x = np.asarray(x,dtype='<f')
	y = np.asarray(y,dtype='<f')
	
	u=(u+1)*(2.2)/2-1.1
	
	dataid=saveResult
	d = getBData_2D(args.data_pathLow+'RESU'+'{:04d}'.format(dataid)+'.DAT',args.dim)		

	du = resize(d[0],(args.dimHigh[0],args.dimHigh[1]),order=3)

	du = du.flatten('F')
	resultu=du+u
		
	outputFilePath1='./logs/{:04d}'.format(dataid)+'.plt'
	outputFilePath='./logs/Delta{:04d}'.format(dataid)+'.plt'
	coords = get_mgrid([args.dimHigh[0],args.dimHigh[1]],dim=2,t=0)
	saveFile2Duw(outputFilePath1,resultu,coords,args.dimHigh)
	saveFile2Duw(outputFilePath,u,coords,args.dimHigh)
	

	return epoch_loss

def train_epoch(model, dataloader,opt,criterion,scheduler,args,n):

	epoch_loss  = 0
	print("begin _train_epoch")
	# with autograd.detect_anomaly():
	model=model.cuda()
	for idx, batch in enumerate(dataloader):
		batch_t1,batch_t2,batch_t3, batch_x, batch_y, batch_data= batch
		batch_t1 = batch_t1.cuda()
		batch_t2 = batch_t2.cuda()
		batch_t3 = batch_t3.cuda()
		batch_x = batch_x.cuda()
		batch_y = batch_y.cuda()
		
		batch_data = batch_data.cuda()
		
		
		yhat = model(batch_x,batch_y,batch_t1,batch_t2,batch_t3)
		loss = criterion(yhat.view(-1), batch_data.view(-1))
		opt.zero_grad()
		loss.backward()
		opt.step()

		epoch_loss +=loss.data.mean().item()
		

	print("end _train_epoch")
	return epoch_loss


def trainNet(model,dataloader_train, args,scheduler, validation_data, save_model_freq):
	model.load_state_dict(torch.load('logs/1000.pth'))
	optim = torch.optim.AdamW(model.parameters(),lr=1e-6)

	criterion = nn.MSELoss(reduction='mean')
	start_time = datetime.datetime.now()
	
	loss = open('./loss1.txt','w')
	loss_track = []
	scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optim,mode='min',factor=0.1,patience=20,verbose=True,threshold=0.0001,threshold_mode='rel',cooldown=0,eps=1e-9)
	for n in range(1001,args.epochs+1):

		t_loss = train_epoch(model,dataloader_train,optim,criterion,scheduler,args,n)
		
		print(f'Train one epoch in {str(datetime.datetime.now() - start_time).split(".")[0]}')
		v_loss = 0
		if n%50==0:
			saveResult=True
			with torch.no_grad():
				v_loss = validate(args,model,criterion,validation_data,saveResult)
		print(n," train_loss: ",t_loss," val_loss: ",v_loss)
		loss.write("Epochs "+str(n)+": train_loss = "+str(t_loss)+": val_loss = "+str(v_loss))
		loss.write('\n')
		if n % 200 == 0 :
			torch.save(model.state_dict(),'logs/'+str(n)+'.pth')
		scheduler.step(t_loss)

	loss.close()
	print(f'Training completed in {str(datetime.datetime.now() - start_time).split(".")[0]}')
def infNet(model, args,saveResult, validation_data):
	model.load_state_dict(torch.load('logs/2000.pth'))

	criterion = nn.MSELoss(reduction='mean')
	start_time = datetime.datetime.now()
	with torch.no_grad():
		v_loss = validate(args,model,criterion,validation_data,saveResult)
		print(" val_loss: ",v_loss)
	
	print(f'val completed in {str(datetime.datetime.now() - start_time).split(".")[0]}')
