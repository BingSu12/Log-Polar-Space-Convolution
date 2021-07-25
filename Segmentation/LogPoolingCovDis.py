import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from torchvision import models, datasets, transforms
#from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from torch.autograd.function import once_differentiable
from torch.autograd.gradcheck import gradcheck
import os


class LogPoolingCovLayer(torch.nn.Module):
    # defind hyper-parameters
    def __init__(self, h, w, stride=2, pool_type='avg_pool', num_levels=2, ang_levels=8, center=[], ini_angle=0, ratio=-1, facbase=2):
        super(LogPoolingCovLayer, self).__init__()
        self.center = center
        self.ini_angle = ini_angle
        self.num_levels = num_levels
        self.ang_levels = ang_levels
        self.pool_type = pool_type
        self.ratio = ratio
        self.stride = stride
        self.h = h
        self.w = w

        if len(center) == 0:
            center = [(h-1)//2, (w-1)//2]
        if ratio<=0:
            ratio = h/w

        ang_indicators = torch.zeros(ang_levels,dtype=torch.float).cuda()
        for ang_count in range(ang_levels):
            ang_indicators[ang_count] = 2*math.pi*(ang_count+1)/ang_levels - 1e-6

        angmap = torch.zeros(h,w,dtype=torch.long).cuda()
        posmap = torch.zeros(h,w,dtype=torch.float).cuda()
        
        for h_count in range(h):
            for w_count in range(w):
                ypos = h_count - center[0]
                xpos = w_count - center[1]
                cur_ang = 0
                if xpos==0:
                    if ypos>0:
                        cur_ang = -math.pi/2
                    if ypos<0:
                        cur_ang = math.pi/2
                    if ypos==0:
                        cur_ang = 0
                else:
                    if ypos==0:
                        if xpos<0:
                            cur_ang = math.pi
                        if xpos>0:
                            cur_ang = 0
                    else:
                        cur_ang = -math.atan(ypos/xpos)
                        if xpos<0:
                            if cur_ang<=0:
                                cur_ang = math.pi + cur_ang
                            else:
                                cur_ang = cur_ang - math.pi

                cur_ang = cur_ang + ini_angle
                if cur_ang>2*math.pi:
                    cur_ang = cur_ang - 2*math.pi
                if cur_ang<0:
                    cur_ang = 2*math.pi + cur_ang
                ang_count = 0
                
                while (ang_count<ang_levels):
                    if  cur_ang>ang_indicators[ang_count].item():
                        ang_count = ang_count + 1
                    else:
                        angmap[h_count,w_count] = ang_count
                        break;
                if ang_count==ang_levels:
                    angmap[h_count,w_count] = ang_count - 1
                
                posmap[h_count,w_count] = math.pow(xpos*math.cos(ini_angle)-ypos*math.sin(ini_angle), 2) + math.pow(-xpos*math.sin(ini_angle)-ypos*math.cos(ini_angle),2)/math.pow(ratio,2)

        
        
        if h>w:
            dmax = (h-1)//2
        else:
            dmax = (w-1)//2
        dmax = np.power(dmax,2)
        facbase = np.power(facbase,2)
        dfac = np.power(facbase,(num_levels-1))
        d1 = dmax/dfac
        if d1<2:
            d1 = 2
        dis_indicator = np.zeros(num_levels,dtype=np.float32)
        for dis_count in range(num_levels):
            dis_indicator[dis_count] = d1*np.power(facbase,dis_count)
        
        print(dmax,dis_indicator)
        
        poslevelmap = torch.zeros(h,w,dtype=torch.long).cuda()
        for h_count in range(h):
            for w_count in range(w):
                pos_count = 0
                while (pos_count<num_levels):
                    if  posmap[h_count,w_count].item()>dis_indicator[pos_count]:
                        pos_count = pos_count + 1
                    else:
                        poslevelmap[h_count,w_count] = pos_count
                        break
                if pos_count==num_levels:
                    poslevelmap[h_count,w_count] = pos_count-1

        self.poslevelmap = poslevelmap
        self.angmap = angmap

        targetmap = torch.zeros(h,w,dtype=torch.long).cuda()
        index_count = 0
        nl_count = -1        
        for nl_target in range(self.num_levels-1,-1,-1):
            nl_count = nl_count + 1
            al_count = -1
            for al_target in range(int(self.ang_levels/2)-1,-1,-1):
                al_count = al_count + 1
                index_count = index_count + 1
                mask = torch.eq(self.poslevelmap,nl_target) & torch.eq(self.angmap,al_target)
                targetmap[mask] = index_count
        
        for nl_target in range(self.num_levels):
            nl_count = nl_count + 1
            al_count = -1
            for al_target in range(int(self.ang_levels/2),self.ang_levels):
                al_count = al_count + 1
                index_count = index_count + 1
                mask = torch.eq(self.poslevelmap,nl_target) & torch.eq(self.angmap,al_target)
                targetmap[mask] = index_count

        print(targetmap)  
        self.center = center      
        self.targetmap = targetmap



    def forward(self, x):
        hwin = self.h
        wwin = self.w
        num, c, h, w = x.size()
        
        hpadval = (hwin-1)//2
        wpadval = (wwin-1)//2
        x_unf = F.unfold(x, (hwin, wwin), padding=(hpadval,wpadval), stride=self.stride)
        
        out_h = (h + 2*hpadval - hwin) // self.stride + 1
        out_w = (w + 2*wpadval - wwin) // self.stride + 1
        
        x_unf = x_unf.view(num,c,hwin,wwin,out_h,out_w).permute(0,1,4,5,2,3).contiguous()
        targetmap = self.targetmap
        tempact = torch.zeros(num,c,out_h,out_w,self.num_levels*2,int(self.ang_levels/2),dtype=torch.float).cuda()

        nl_count = -1 
        index_count = 0       
        for nl_target in range(self.num_levels-1,-1,-1):
            nl_count = nl_count + 1
            al_count = -1
            for al_target in range(int(self.ang_levels/2)-1,-1,-1):
                al_count = al_count + 1
                index_count = index_count + 1
                mask = torch.eq(targetmap,index_count).cuda()
                mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(x_unf.size())
                pc = torch.masked_select(x_unf,mask).view(num,c,out_h,out_w,-1)
                
                if pc.shape[-1]>0:
                    if self.pool_type == 'max_pool':
                        tempact[:,:,:,:,nl_count,al_count] = torch.max(pc,-1).values
                    if self.pool_type == 'sum_pool':
                        tempact[:,:,:,:,nl_count,al_count] = torch.sum(pc,-1,True).view(num,c,out_h,out_w)
                    if self.pool_type == 'avg_pool':
                        tempact[:,:,:,:,nl_count,al_count] = torch.mean(pc,-1,True).view(num,c,out_h,out_w)
        

        for nl_target in range(self.num_levels):
            nl_count = nl_count + 1
            al_count = -1
            for al_target in range(int(self.ang_levels/2),self.ang_levels):
                al_count = al_count + 1
                index_count = index_count + 1
                mask = torch.eq(targetmap,index_count).cuda()
                mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(x_unf.size())
                pc = torch.masked_select(x_unf,mask).view(num,c,out_h,out_w,-1)
                if pc.shape[-1]>0:
                    if self.pool_type == 'max_pool':
                        tempact[:,:,:,:,nl_count,al_count] = torch.max(pc,-1).values
                    if self.pool_type == 'sum_pool':
                        tempact[:,:,:,:,nl_count,al_count] = torch.sum(pc,-1,True).view(num,c,out_h,out_w)
                    if self.pool_type == 'avg_pool':
                        tempact[:,:,:,:,nl_count,al_count] = torch.mean(pc,-1,True).view(num,c,out_h,out_w)

        tempact = tempact.permute(0,1,4,5,2,3).contiguous().view(num,-1,out_h*out_w)
        output = F.fold(tempact, (out_h*self.num_levels*2, out_w*int(self.ang_levels/2)), (self.num_levels*2,int(self.ang_levels/2)), padding=0, stride=(self.num_levels*2,int(self.ang_levels/2)))
        return output


        
            




if __name__ == '__main__':
    print(torch.__version__)
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    dtype = torch.float
    device = torch.device("cuda")
    pool_type = 'max_pool'
    num_levels = 2
    ang_levels = 8
    # device = torch.device("cuda:0") # Uncomment this to run on GPU

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, W = 2, 3, 32, 32
    Stride = 1
    hwin, wwin = 9, 9
    out_h = (H - 1) // Stride + 1
    out_w = (W - 1) // Stride + 1

    # Create random Tensors to hold input and outputs.
    x = torch.randn(N, D_in, H, W, device=device, dtype=dtype, requires_grad=True)
    #x = torch.zeros(N, D_in, H, W, device=device, dtype=dtype, requires_grad=True)+1
    y = torch.randn(N, D_in, num_levels*2*out_h, int(ang_levels/2)*out_w, device=device, dtype=dtype, requires_grad=False)
    

    LogPL2 = LogPoolingCovLayer(5, 5, stride=Stride, pool_type=pool_type, num_levels=2, ang_levels=6, facbase=2)
    LogPL2 = LogPoolingCovLayer(5, 5, stride=Stride, pool_type=pool_type, num_levels=2, ang_levels=6, facbase=3)
    LogPL3 = LogPoolingCovLayer(9, 9, stride=Stride, pool_type=pool_type, num_levels=2, ang_levels=6, facbase=3)
    LogPL3 = LogPoolingCovLayer(9, 9, stride=Stride, pool_type=pool_type, num_levels=2, ang_levels=8, facbase=2)

    