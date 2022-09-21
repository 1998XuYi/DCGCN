# -*- coding: utf-8 -*-
"""
Created on 2022.9.20

@author: Xu Yi
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter,LayerNorm,InstanceNorm2d

from utils import ST_BLOCK_0
from utils import DGL_BLOCK

"""
the parameters:
x-> [batch_num,in_channels,num_nodes,tem_size],
"""
# c_in=num_of_features, c_out=64,
#                 num_nodes=num_nodes, week=24,
#                 K=3, Kt=3
class DCGCN_block(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,K,Kt):
        super(DCGCN_block,self).__init__()
        self.block1=ST_BLOCK_0(c_in,c_out,num_nodes,tem_size,K,Kt)
        self.block2=ST_BLOCK_0(c_out,c_out,num_nodes,tem_size,K,Kt)
        self.final_conv=Conv2d(tem_size,12,kernel_size=(1, c_out),padding=(0,0),
                          stride=(1,1), bias=True)
        self.w=Parameter(torch.zeros(num_nodes,12), requires_grad=True)
        nn.init.xavier_uniform_(self.w)

    def forward(self,x,supports):
        # print("123456 x.shape1", x.shape)
        x,_,_ = self.block1(x,supports)
        # print("123456 x.shape2", x.shape)
        x,d_adj,t_adj = self.block2(x,supports)
        # print("123456 x.shape3",x.shape)
        x = x.permute(0,3,2,1)
        # print("123456 x.shape4", x.shape)
        # print("self.final_conv(x).shape",self.final_conv(x).shape)
        # print("self.final_conv(x).squeeze().shape", self.final_conv(x).squeeze().shape)
        x = self.final_conv(x).squeeze().permute(0,2,1)#b,n,12
        x = x*self.w
        return x,d_adj,t_adj


# c_in=num_of_features, c_out=64,
#                 num_nodes=num_nodes, week=24,
#                 day=12, recent=24,
#                 K=3, Kt=3
class DCGCN(nn.Module):
    # week=12,day=12,recent=36
    def __init__(self,c_in,c_out,num_nodes,week,day,recent,K,Kt):
        tem_size = week + day + recent
        super(DCGCN,self).__init__()
        self.block_w=DCGCN_block(c_in,c_out,num_nodes,week,K,Kt)
        self.block_d=DCGCN_block(c_in,c_out,num_nodes,day,K,Kt)
        self.block_r=DCGCN_block(c_in,c_out,num_nodes,recent,K,Kt)
        self.bn=BatchNorm2d(c_in,affine=False)
        self.h = Parameter(torch.zeros(num_nodes, num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        self.DGL_BLOCK = DGL_BLOCK(c_in, c_out, num_nodes, tem_size, K, Kt)
    def forward(self,x_w,x_d,x_r,supports):
        x_w=self.bn(x_w)
        x_d=self.bn(x_d)
        x_r=self.bn(x_r)
        A = self.h + supports
        d = 1 / (torch.sum(A, -1) + 0.0001)
        D = torch.diag_embed(d)
        A = torch.matmul(D, A)
        supports = F.dropout(A, 0.5, self.training)
        x=torch.cat((x_w,x_d,x_r),-1)
        supports = self.DGL_BLOCK(x, supports)
        x_w,_,_=self.block_w(x_w,supports)
        x_d,_,_=self.block_d(x_d,supports)
        x_r,d_adj_r,t_adj_r=self.block_r(x_r,supports)
        out=x_w+x_d+x_r
        return out,d_adj_r,t_adj_r
    

    
    
    
    
    