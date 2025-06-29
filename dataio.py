import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
from skimage.transform import resize
from skimage.io import imread, imsave
from skimage import data,img_as_float
import torch.nn.functional as F
import struct
import math
from sklearn.metrics.pairwise import rbf_kernel

def saveFile2D(outputFilePath,data,coords,dim):

	print(coords.shape)

	f=open(outputFilePath,"w")
	
	f.write(" variables=\"x\",\"y\",\"u\" \n" )
	f.write("zone i= "+ '{:4d}'.format(dim[0])+" ,j= "+'{:4d}\n'.format(dim[1]))
	index=0
	for j in range(dim[1]):
		for i in range(dim[0]):
			f.write("%.7f %.7f %.7f\n"%(float(coords[index][2]),float(coords[index][1]),float(data[index])))
			index+=1
		
	f.close()
def saveFile2Dxy(outputFilePath,data,x,y,dim):

	f=open(outputFilePath,"w")
	
	f.write(" variables=\"x\",\"y\",\"u\" \n" )
	f.write("zone i= "+ '{:4d}'.format(dim[0])+" ,j= "+'{:4d}\n'.format(dim[1]))
	index=0
	for j in range(dim[1]):
		for i in range(dim[0]):
			f.write("%.7f %.7f %.7f\n"%(float(x[index]),float(y[index]),float(data[index])))
			index+=1
		
	f.close()
def getDeltaData1(dataPath,dim):
	value=np.zeros([4,(dim[0]),(dim[1])])
	index=0
	with open(dataPath,'r') as f:
		for j in range(dim[1]):
			for i in range(dim[0]):
				line=f.readline().strip().split()
				if math.isnan(float(line[0])):
					print(dataPath," ",i," ",j)
				value[0,i,j]=float(line[0])
				value[1,i,j]=float(line[1])
				value[2,i,j]=float(line[2])
				value[3,i,j]=float(line[3])
				index+=1
	return value
	
def getCoor5Point_3dt(dim,deltad,t1,t2,t3):
    valuet1=np.zeros((dim[0]*dim[1],4,1))
    valuet2=np.zeros((dim[0]*dim[1],4,1))
    valuet3=np.zeros((dim[0]*dim[1],4,1))
    valuex=np.zeros((dim[0]*dim[1],4,1))
    valuey=np.zeros((dim[0]*dim[1],4,1))
    valued=np.zeros((dim[0]*dim[1],4,1))
    # valuev=np.zeros((dim[0]*dim[1],4,1))
    index = 0
    valuet1.fill(t1)
    valuet2.fill(t2)
    valuet3.fill(t3)
    xinter=0.25/dim[0]
    yinter=1.0/dim[1]
    for j in range(dim[1]):
        for i in range(dim[0]):
            valuex[index,0,0]=i*xinter
            valuex[index,1,0]=(i-1)*xinter
            valuex[index,2,0]=(i-1)*xinter
            valuex[index,3,0]=(i+1)*xinter
            # valuex[index,4,0]=(i+1)*xinter
            
            valuey[index,0,0]=j*yinter
            valuey[index,1,0]=(j-1)*yinter
            valuey[index,2,0]=(j+1)*yinter
            valuey[index,3,0]=(j-1)*yinter
            # valuey[index,4,0]=(j+1)*yinter
            
            if dim[0]-1>i>0 and dim[1]-1>j>0:
            
                valued[index,0,0]=deltad[0][j*dim[0]+i]
                valued[index,1,0]=deltad[0][(j-1)*dim[0]+(i-1)]
                valued[index,2,0]=deltad[0][(j+1)*dim[0]+(i-1)]
                valued[index,3,0]=deltad[0][(j-1)*dim[0]+(i+1)]
            else:
                valued[index,0,0]=deltad[0][j*dim[0]+i]
                valued[index,1,0]=deltad[0][j*dim[0]+i]
                valued[index,2,0]=deltad[0][j*dim[0]+i]
                valued[index,3,0]=deltad[0][j*dim[0]+i]
            index+=1
    return valuet1,valuet2,valuet3,valuex,valuey,valued
def getCoor5PointChinaVis(dim,deltad,ti):
    valuet=np.zeros((dim[0]*dim[1],4,1))
    valuex=np.zeros((dim[0]*dim[1],4,1))
    valuey=np.zeros((dim[0]*dim[1],4,1))
    valued=np.zeros((dim[0]*dim[1],4,1))
    index = 0
    valuet.fill((ti-460)*0.01/40)
    xinter=0.25/dim[0]
    yinter=1.0/dim[1]
    for j in range(dim[1]):
        for i in range(dim[0]):
            valuex[index,0,0]=i*xinter
            valuex[index,1,0]=(i-1)*xinter
            valuex[index,2,0]=(i-1)*xinter
            valuex[index,3,0]=(i+1)*xinter
            
            valuey[index,0,0]=j*yinter
            valuey[index,1,0]=(j-1)*yinter
            valuey[index,2,0]=(j+1)*yinter
            valuey[index,3,0]=(j-1)*yinter
            
            if dim[0]-1>i>0 and dim[1]-1>j>0:
            
                valued[index,0,0]=deltad[0][j*dim[0]+i]
                valued[index,1,0]=deltad[0][(j-1)*dim[0]+(i-1)]
                valued[index,2,0]=deltad[0][(j+1)*dim[0]+(i-1)]
                valued[index,3,0]=deltad[0][(j-1)*dim[0]+(i+1)]
            else:
                valued[index,0,0]=deltad[0][j*dim[0]+i]
                valued[index,1,0]=deltad[0][j*dim[0]+i]
                valued[index,2,0]=deltad[0][j*dim[0]+i]
                valued[index,3,0]=deltad[0][j*dim[0]+i]
            index+=1
    return valuet,valuex,valuey,valued
def getCoor5Point(dim,deltad,ti):
    valuet=np.ones((dim[0]*dim[1],5,1))
    valuex=np.zeros((dim[0]*dim[1],5,1))
    valuey=np.zeros((dim[0]*dim[1],5,1))
    valueu=np.zeros((dim[0]*dim[1],5,1))
    index = 0
    
    xinter=0.25/dim[0]
    yinter=1.0/dim[1]
    for j in range(dim[1]):
        for i in range(dim[0]):
            valuex[index,0,0]=i*xinter
            valuex[index,1,0]=(i-1)*xinter
            valuex[index,2,0]=(i-1)*xinter
            valuex[index,3,0]=(i+1)*xinter
            valuex[index,4,0]=(i+1)*xinter
            
            valuey[index,0,0]=j*yinter
            valuey[index,1,0]=(j-1)*yinter
            valuey[index,2,0]=(j+1)*yinter
            valuey[index,3,0]=(j-1)*yinter
            valuey[index,4,0]=(j+1)*yinter
            
            if dim[0]-1>i>0 and dim[1]-1>j>0:
            
                valueu[index,0,0]=deltad[j*dim[0]+i]
                valueu[index,1,0]=deltad[(j-1)*dim[0]+(i-1)]
                valueu[index,2,0]=deltad[(j+1)*dim[0]+(i-1)]
                valueu[index,3,0]=deltad[(j-1)*dim[0]+(i+1)]
                valueu[index,4,0]=deltad[(j+1)*dim[0]+(i+1)]
            else:
                valueu[index,0,0]=deltad[j*dim[0]+i]
                valueu[index,1,0]=deltad[j*dim[0]+i]
                valueu[index,2,0]=deltad[j*dim[0]+i]
                valueu[index,3,0]=deltad[j*dim[0]+i]
                valueu[index,4,0]=deltad[j*dim[0]+i]
            index+=1
    return valuet,valuex,valuey,valueu
def getBData_2D(dataPath,dim):
	value=np.zeros([4,(dim[0]),(dim[1])])
	index=0
	res=open(dataPath,'rb')
	for j in range(dim[1]+1):
		for i in range(dim[0]+1):
			_=res.read(4)
			n1=res.read(4)
			n2=res.read(4)
			n3=res.read(4)
			n4=res.read(4)
			_=res.read(4)
			if i!=dim[0] and j !=dim[1]:
				v1=struct.unpack('1f',n1)
				v2=struct.unpack('1f',n2)
				v3=struct.unpack('1f',n3)
				v4=struct.unpack('1f',n4)
				
				value[0,i,j]=float(v1[0])
				value[1,i,j]=float(v2[0])
				value[2,i,j]=float(v3[0])
				value[3,i,j]=float(v4[0])
			
	res.close()
	return value
def getDeltaData(dataPath,dim):
	value=np.zeros([4,(dim[0])*(dim[1])])
	index=0
	with open(dataPath,'r') as f:
		for j in range(dim[1]):
			for i in range(dim[0]):
				line=f.readline().strip().split()
				if math.isnan(float(line[0])):
					print(dataPath," ",i," ",j)
				value[0,index]=float(line[0])
				value[1,index]=float(line[1])
				value[2,index]=float(line[2])
				value[3,index]=float(line[3])
				index+=1
	return value

def get_txy_Vordata(dataPath,dim):
	value=np.zeros([dim,4])
	with open(dataPath,'r') as f:
		for i in range(dim):
			line=f.readline().strip().split()
			value[i,0]=float(line[0])
			value[i,1]=float(line[1])
			value[i,2]=float(line[2])
			value[i,3]=float(line[3])
	return value
	
def get_txy_Vordata5Point(dataPath):
	with open(dataPath,'r') as f:
		line=f.readline().strip().split()
		dim=int(line[0])
		# print(dim)
		value=np.zeros([dim//10,5,4,1])
		for i in range(dim//10):
			for j in range(5):
				line=f.readline().strip().split()
				value[i,j,0,0]=float(line[0])
				value[i,j,1,0]=float(line[1])
				value[i,j,2,0]=float(line[2])
				value[i,j,3,0]=float(line[3])
			for _ in range(5):
				line=f.readline().strip().split()
	return value

class RT3_16X(Dataset) : 
	def __init__(self,args,start=0, end=2000, train=False):
		super().__init__()
		self.mode = 'train'*train + 'val'*(1-train)
		dataD = []
		
		datax = []
		datay = []
		# dataz = []
		datat1 = []
		datat2 = []
		datat3 = []
		num_samples = 32000
		print("load data")
		tarray = np.arange(0,101).reshape(-1,1)
		sigma=2000.0
		K = rbf_kernel(tarray,gamma=1/(2*sigma**2))
		from sklearn.decomposition import KernelPCA
		kpca = KernelPCA(kernel='precomputed',n_components=3)
		datat_3d = kpca.fit_transform(K)
		datat_3d = (datat_3d - np.min(datat_3d)) / (np.max(datat_3d) - np.min(datat_3d))+0.01
		datat_3d = np.log(datat_3d)
		for i in range(start,end):
			data=getDeltaData(args.data_pathHigh+'Deltafull'+'{:04d}'.format(i)+'.DAT',args.dimHigh)
			dataCoor9Point = getCoor5Point_3dt(args.dimHigh,data,datat_3d[i-400,0]*(-0.1),datat_3d[i-400,1]*(-1),datat_3d[i-400,2]*(-1))
			if train ==True:
				indices = np.random.choice(args.dimHigh[0]*args.dimHigh[1], num_samples, replace=False)
				datat1 += list(dataCoor9Point[0][indices])
				datat2 += list(dataCoor9Point[1][indices])
				datat3 += list(dataCoor9Point[2][indices])
				datax += list(dataCoor9Point[3][indices])
				datay += list(dataCoor9Point[4][indices])
				dataD += list(dataCoor9Point[5][indices])
				
			else:
				datat1 += list(dataCoor9Point[0])
				datat2 += list(dataCoor9Point[1])
				datat3 += list(dataCoor9Point[2])
				datax += list(dataCoor9Point[3])
				datay += list(dataCoor9Point[4])
				dataD += list(dataCoor9Point[5])
	
		dataD = np.asarray(dataD)
		
		dataD = 2*(dataD-(-1.1))/(1.1-(-1.1))-1
		
		datat1 = np.asarray(datat1)
		datat2 = np.asarray(datat2)
		datat3 = np.asarray(datat3)
		datax = np.asarray(datax)
		
		datay = np.asarray(datay)
		
		self.datat1   = torch.tensor(datat1).float()
		self.datat2   = torch.tensor(datat2).float()
		self.datat3   = torch.tensor(datat3).float()
		self.datax   = torch.tensor(datax).float()
		self.datay   = torch.tensor(datay).float()
		self.dataD = torch.tensor(dataD).float()
		print("load end")
	
	def __getitem__(self ,index):
		return self.datat1[index, :],self.datat2[index, :],self.datat3[index, :], self.datax[index,:],self.datay[index, :], self.dataD[index,:]
		
	def __len__(self):
		return self.datax.shape[0]
