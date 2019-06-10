import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

'''Residual block --
       Refererence - http://torch. ch/blog/2016/02/04/resnets.html
'''
class res_block(nn.Module):
    def __init__(self, C):
        super(res_block, self).__init__()
        self.conv1 = nn.Conv2d(C, C, 3, padding=1)
        self.conv2 = nn.Conv2d(C, C, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(C)
        self.bn2 = nn.BatchNorm2d(C)
        
    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = h + x
        y = F.relu(h)
        
        return y
    
'''Non-Residual block --
        consists of a convolutional layer, a batch normalization layer and a relu/tanh layer
'''
class non_res_block(nn.Module):
    def __init__(self, C_in, C_out, k=3, mode='normal'):
        super(non_res_block, self).__init__()
        self.conv = nn.Conv2d(C_in, C_out, k, padding=np.int((k-1)/2))
        self.bn = nn.BatchNorm2d(C_out)
        self.mode = mode
        
    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h)
        if self.mode == 'normal':
            y = F.relu(h)
        else:
            y = torch.tanh(h)
                
        return y

'''Transformation network --
       Architecture: 1. Three up sampling blocks to scale the features to 128. 
                        The first one has a kernal size of 9, others of 3.
                     2. 5 residual blocks with convolutional kernal size of 3.
                     3. Three down sampling blocks to decrease the features back to 3.
                        The last one has a kernal size of 9 with tanh as activation function
       Reference - https://arxiv.org/pdf/1603.08155.pdf
'''
class TransNet(nn.Module):
    def __init__(self, D=5):
        super(TransNet, self).__init__()
        self.D = D
        self.up1 = non_res_block(3, 32, k=9)
        self.up2 = non_res_block(32, 64)
        self.up3 = non_res_block(64, 128)
        self.res = nn.ModuleList()
        for ii in range(self.D):
            self.res.append(non_res_block(128, 128))
        self.dn1 = non_res_block(128, 64)
        self.dn2 = non_res_block(64, 32)
        self.dn3 = non_res_block(32, 3, k=9, mode='last')
        
        for param in self.parameters():
            param.requires_grad = True
        
    def forward(self, x): 
        h = self.up1(x)
        h = self.up2(h)
        h = self.up3(h)
        
        for ii in range(self.D):
            h = self.res[ii](h)
            
        h = self.dn1(h)
        h = self.dn2(h)
        y = self.dn3(h)
        
        return y