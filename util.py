import numpy as np
import torch.nn as nn
import copy
import torch
from torch.nn import init

	
def nifll_scheduler(epoch, lr):
       if epoch < 200:
           return lr
       elif epoch < 400:
           return 1e-4
       else:
           return 1e-5

def weights_init_kaiming(m):
	classname = m.__class__.__name__
	print("classname: ",classname)
	if classname.find("Conv")!=-1:
		init.kaiming_uniform_(m.weight.data)
	elif classname.find("Linear")!=-1:
		init.kaiming_uniform_(m.weight.data)
	elif classname.find("BatchNorm")!=-1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def weights_init_normal(m):
	classname = m.__class__.__name__
	if classname.find("Conv")!=-1:
		init.normal_(m.weight.data, 1.0, 0.02)
	elif classname.find("Linear")!=-1:
		init.normal_(m.weight.data, 1.0, 0.02)
	elif classname.find("BatchNorm")!=-1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)
def init_weights_truncNorm(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.trunc_normal_(m.weight, std=0.1)
        torch.nn.init.trunc_normal_(m.bias, std=0.1)
