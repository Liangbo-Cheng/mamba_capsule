import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pdb
from torch.autograd.variable import Variable
from torch.nn.modules.loss import _Loss
from einops import rearrange
from .vmamba_mcr import *


class PrimaryCaps_semantic(nn.Module):
    def __init__(self, c1=32, c2=8, K=1, P=4, stride=1):
        super(PrimaryCaps_semantic, self).__init__()
        self.pose = nn.Conv2d(in_channels=c1, out_channels=c2*P*P,
                            kernel_size=K, stride=stride, padding=1, bias=True)
        self.a = nn.Conv2d(in_channels=c1, out_channels=c2,
                            kernel_size=K, stride=stride, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):                
        p = self.pose(x)                 
        a = self.a(x)                   
        a = self.sigmoid(a)
        out = torch.cat([p, a], dim=1)    
        return out, p, a

class ConvCaps_semantic(nn.Module):
    def __init__(self, B=8, C=8, K=1, P=4, stride=2, iters=3,
                 coor_add=False, w_shared=False, horizontal=False, vertical=False):
        super(ConvCaps_semantic, self).__init__()
        # TODO: lambda scheduler
        # Note that .contiguous() for 3+ dimensional tensors is very slow
        self.horizontal = horizontal
        self.vertical = vertical
        self.B = B
        self.C = C
        self.K = K
        self.P = P
        self.psize = P*P
        self.stride = stride
        self.iters = iters
        self.coor_add = coor_add
        self.w_shared = w_shared
        # constant
        self.eps = 1e-8
        self._lambda = 1e-03
        self.ln_2pi = torch.FloatTensor(1).fill_(math.log(2*math.pi))    
        # params
        # Note that \beta_u and \beta_a are per capsule type,
        # which are stated at https://openreview.net/forum?id=HJWLfGWRb&noteId=rJUY2VdbM
        self.beta_u = nn.Parameter(torch.zeros(C))
        self.beta_a = nn.Parameter(torch.zeros(C))
        # Note that the total number of trainable parameters between
        # two convolutional capsule layer types is 4*4*k*k
        # and for the whole layer is 4*4*k*k*B*C,
        # which are stated at https://openreview.net/forum?id=HJWLfGWRb&noteId=r17t2UIgf
        self.weights = nn.Parameter(torch.randn(1, K*K*B, C, P, P))
        # op
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)

    def m_step(self, a_in, r, v, eps, b, B, C, psize):

        r = r * a_in
        r = r / (r.sum(dim=2, keepdim=True) + eps)
        r_sum = r.sum(dim=1, keepdim=True)
        coeff = r / (r_sum + eps)
        coeff = coeff.view(b, B, C, 1)

        mu = torch.sum(coeff * v, dim=1, keepdim=True)
        sigma_sq = torch.sum(coeff * (v - mu)**2, dim=1, keepdim=True) + eps

        r_sum = r_sum.view(b, C, 1)
        sigma_sq = sigma_sq.view(b, C, psize)
        cost_h = (self.beta_u.view(C, 1) + torch.log(sigma_sq.sqrt())) * r_sum

        a_out = self.sigmoid(self._lambda*(self.beta_a - cost_h.sum(dim=2)))
        sigma_sq = sigma_sq.view(b, 1, C, psize)

        return a_out, mu, sigma_sq

    def e_step(self, mu, sigma_sq, a_out, v, eps, b, C):
        ln_p_j_h = -1. * (v - mu)**2 / (2 * sigma_sq) \
                    - torch.log(sigma_sq.sqrt()) \
                    - 0.5*self.ln_2pi

        ln_ap = ln_p_j_h.sum(dim=3) + torch.log(a_out.view(b, 1, C))
        r = self.softmax(ln_ap)
        return r

    def caps_em_routing(self, v, a_in, C, eps):
        b, B, c, psize = v.shape       
        assert c == C
        assert (b, B, 1) == a_in.shape     

        r = torch.FloatTensor(b, B, C).fill_(1./C)  

        if a_in.is_cuda and a_in.get_device() == 0:
            device = torch.device("cuda:0")
            r=r.to(device)
            self.ln_2pi=self.ln_2pi.to(device)  
        elif a_in.is_cuda and a_in.get_device() == 1:
            device = torch.device("cuda:1")
            r=r.to(device)
            self.ln_2pi=self.ln_2pi.to(device)   
        elif a_in.is_cuda and a_in.get_device() == 2:
            device = torch.device("cuda:2")
            r=r.to(device)
            self.ln_2pi=self.ln_2pi.to(device)   
        elif a_in.is_cuda and a_in.get_device() == 3:
            device = torch.device("cuda:3")
            r=r.to(device)
            self.ln_2pi=self.ln_2pi.to(device)   
    
        for iter_ in range(self.iters):
            a_out, mu, sigma_sq = self.m_step(a_in, r, v, eps, b, B, C, psize)
            if iter_ < self.iters - 1:
                r = self.e_step(mu, sigma_sq, a_out, v, eps, b, C)

        return mu, a_out

    def add_pathes(self, x, B, K, psize, stride):
        """
            Shape:
                Input:     (b, H, W, B*(P*P+1))
                Output:    (b, H', W', K, K, B*(P*P+1))
        """
        b, h, w, c = x.shape      
        assert h == w         
        assert c == B*(psize+1)   
        oh = ow = 1 
        x = torch.unsqueeze(x, dim=1)
        x = torch.unsqueeze(x, dim=3)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()      
        return x, oh, ow

    def transform_view(self, x, w, C, P, w_shared=False):
        """
            For conv_caps:
                Input:     (b*H*W, K*K*B, P*P)
                Output:    (b*H*W, K*K*B, C, P*P)
            For class_caps:
                Input:     (b, H*W*B, P*P)
                Output:    (b, H*W*B, C, P*P)
        """
        b, B, psize = x.shape           
        assert psize == P*P

        x = x.view(b, B, 1, P, P)        
        if w_shared:
            hw = int(B / w.size(1))
            w = w.repeat(1, hw, 1, 1, 1)

        w = w.repeat(b, 1, 1, 1, 1)       
        x = x.repeat(1, 1, C, 1, 1)       
        v = torch.matmul(x, w)             
        v = v.view(b, B, C, P*P)          
        return v        

    def add_coord(self, v, b, h, w, B, C, psize):
        """
            Shape:
                Input:     (b, H*W*B, C, P*P)
                Output:    (b, H*W*B, C, P*P)
        """
        assert h == w
        v = v.view(b, h, w, B, C, psize)
        coor = torch.arange(h, dtype=torch.float32) / h
        coor_h = torch.cuda.FloatTensor(1, h, 1, 1, 1, self.psize).fill_(0.)
        coor_w = torch.cuda.FloatTensor(1, 1, w, 1, 1, self.psize).fill_(0.)
        coor_h[0, :, 0, 0, 0, 0] = coor
        coor_w[0, 0, :, 0, 0, 1] = coor
        v = v + coor_h + coor_w
        v = v.view(b, h*w*B, C, psize)
        return v

    def forward(self, x):        
        x=x.permute(0, 2, 3, 1)    
        b, h, w, c = x.shape             
        if not self.w_shared:
            # add patches
            x, oh, ow = self.add_pathes(x, self.B, self.K, self.psize, self.stride)

            # transform view
            p_in = x[:, :, :, :, :, :self.B*self.psize].contiguous()
            a_in = x[:, :, :, :, :, self.B*self.psize:].contiguous()       
            p_in = p_in.view(b*oh*ow, self.K*self.K*self.B, self.psize)     
            a_in = a_in.view(b*oh*ow, self.K*self.K*self.B, 1)             
            v = self.transform_view(p_in, self.weights, self.C, self.P)     

            v = self.add_coord(v, b, h, w, self.B, self.C, self.psize)

            # em_routing  p:prompt   
            p_out, a_out = self.caps_em_routing(v, a_in, self.C, self.eps) 
            p_out_p = torch.sum(p_out, dim=1)   
            a_out_p = torch.unsqueeze(a_out,dim=2)  
            out_p = torch.cat([p_out_p, a_out_p], dim=2)  

            p_out = p_out.view(b, oh, ow, self.C*self.psize)              
            a_out = a_out.view(b, oh, ow, self.C)                         
            out = torch.cat([p_out, a_out], dim=3)                        
            out = out.permute(0, 3, 1, 2)   
            a_out = a_out.permute(0, 3, 1, 2)
        else:
            assert c == self.B*(self.psize+1)
            assert 1 == self.K
            assert 1 == self.stride
            p_in = x[:, :, :, :self.B*self.psize].contiguous()
            p_in = p_in.view(b, h*w*self.B, self.psize)
            a_in = x[:, :, :, self.B*self.psize:].contiguous()
            a_in = a_in.view(b, h*w*self.B, 1)

            # transform view
            v = self.transform_view(p_in, self.weights, self.C, self.P, self.w_shared)

            # coor_add
            if self.coor_add:
                v = self.add_coord(v, b, h, w, self.B, self.C, self.psize)

            # em_routing
            _, out = self.caps_em_routing(v, a_in, self.C, self.eps)
            _, out = _, out.permute(0, 3, 1, 2)  

        return p_out_p,a_out_p,out_p ,out
    
def autopad(k, p=None, d=1):  
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
    
class ReLUConv(nn.Module):
    default_act = nn.ReLU() 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Semantics(nn.Module):
    def __init__(self, A=256, B=64, C=64, K=3, P=4, iters=3,b=1, w=22,h=22,out_channel=384,feature_size=22):       
        super(Semantics, self).__init__()
        self.A = A 
        self.B = B 
        self.C = C 
        self.K = K 
        self.P = P 
        self.b = b
        self.w = w 
        self.h = h 
        self.feature_size = feature_size
        self.primary_caps_ = PrimaryCaps_semantic(c1=A, c2=B, K=3, P=P, stride=1)
        self.norm2 = nn.LayerNorm(B*(P*P+1))
        self.mlp2 = nn.Sequential(
            nn.Linear(B*(P*P+1), B),
            nn.GELU(),
            nn.Linear(B, B),
        )
        self.mamba = VSSM_cap(dims=(B),ssm_ratio=B/(B))
        self.norm_pose = nn.LayerNorm(P*P)
        self.mlp_pose = nn.Sequential(
            nn.Linear(P*P, P*P),)
        self.norm_a = nn.LayerNorm(P*P)
        self.mlp_a = nn.Sequential(
            nn.Linear(P*P, 1),
            nn.Sigmoid(),)
        self.conv_caps_1 = ConvCaps_semantic(B, C, 1, P, stride=1, iters=iters, horizontal=True, vertical=False)  
        self.conv_caps_2 = ConvCaps_semantic(B, C, 1, P, stride=1, iters=iters, horizontal=True, vertical=False)  
        self.conv_caps_3 = ConvCaps_semantic(B, C, 1, P, stride=1, iters=iters, horizontal=True, vertical=False)  
        self.conv_caps_4 = ConvCaps_semantic(B, C, 1, P, stride=1, iters=iters, horizontal=True, vertical=False)  
        self.sigmoid = nn.Sigmoid()   
        self.conv_keep = ReLUConv(A, A, k=5, s=1, p=2)  
        self.conv = nn.Conv2d(A+B+C*4, out_channel, 1, bias=False) 
        self.bn = nn.BatchNorm2d(out_channel) 
        self.conv_keep_2 = ReLUConv(A, A, k=5, s=1, p=2)  
        self.conv_2 = nn.Conv2d(A+B+C*4, out_channel, 1, bias=False) 
        self.bn_2 = nn.BatchNorm2d(out_channel) 
        self.conv_3 = nn.Conv2d(out_channel*2, out_channel, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn_3 = nn.BatchNorm2d(out_channel) 

    def forward(self, x):       
        caps,pose,act = self.primary_caps_(x)
        #split capsule    
        cut_1 = [[(h_idx + k_idx) \
                for k_idx in range(0, self.B)] \
                for h_idx in range(0, self.P*self.P*self.B, self.B)]   
        cut_2 = [[(h_idx + k_idx) \
                for k_idx in range(0, self.B)] \
                for h_idx in range(0, 1*self.B, self.B)]   
        pose_ = pose[:, cut_1, :, :].contiguous()  
        act_ = act[:, cut_2, :, :].contiguous()    
        Initial =torch.cat([pose_, act_], dim=1)   

        # feature mapping
        Initial_v = rearrange(Initial, " d e f g h->d g h (e f)")
        rgb_fea_1_16_32 = self.mlp2(self.norm2(Initial_v))       
        mamba_y, type_cap_pose  = self.mamba(rgb_fea_1_16_32)     
        mamba_y = rearrange(mamba_y, " a b c d->a d b c")  
        #pose+activation
        type_cap_pose = self.mlp_pose(self.norm_pose(type_cap_pose))
        type_cap_a = self.mlp_a(self.norm_a(type_cap_pose))
        type_cap_pose = rearrange(type_cap_pose, " a b c d e->a b c (d e)")  
        type_cap_a = rearrange(type_cap_a, " a b c d e->a b c (d e)")            
        #for similarity
        type_cap = torch.cat([type_cap_pose, type_cap_a], dim=3)   
        correlation_1_1 = torch.squeeze(type_cap[:,0], dim=1)
        correlation_2_1 = torch.squeeze(type_cap[:,1], dim=1)
        correlation_3_1 = torch.squeeze(type_cap[:,2], dim=1)
        correlation_4_1 = torch.squeeze(type_cap[:,3], dim=1)          
        #for routing
        _b = type_cap_pose.shape[0] 
        prompt_1 = torch.cat([type_cap_pose[:,0].view(_b,-1), type_cap_a[:,0].view(_b,-1)], dim=1)   
        prompt_1 = torch.unsqueeze(prompt_1, dim=2)
        prompt_1 = torch.unsqueeze(prompt_1, dim=2)       

        prompt_2 = torch.cat([type_cap_pose[:,1].view(_b,-1), type_cap_a[:,1].view(_b,-1)], dim=1)  
        prompt_2 = torch.unsqueeze(prompt_2, dim=2)
        prompt_2 = torch.unsqueeze(prompt_2, dim=2)      

        prompt_3 = torch.cat([type_cap_pose[:,2].view(_b,-1), type_cap_a[:,2].view(_b,-1)], dim=1)  
        prompt_3 = torch.unsqueeze(prompt_3, dim=2)
        prompt_3 = torch.unsqueeze(prompt_3, dim=2)      

        prompt_4 = torch.cat([type_cap_pose[:,3].view(_b,-1), type_cap_a[:,3].view(_b,-1)], dim=1)  
        prompt_4 = torch.unsqueeze(prompt_4, dim=2)
        prompt_4 = torch.unsqueeze(prompt_4, dim=2)       

        #Routing1
        p_out, a_out_1, out_1,again = self.conv_caps_1(prompt_1)     
        correlation_1_2 = out_1  
        correlation_1_2 = rearrange(correlation_1_2, " d e f->d f e ")    
        cosin_1 = torch.matmul(correlation_1_1, correlation_1_2)/torch.matmul(torch.norm(correlation_1_1, dim=2,keepdim=True) , torch.norm(correlation_1_2, dim=1,keepdim=True))
        cosin_1 = self.sigmoid(cosin_1)    
        act_1 = act.flatten(2, 3)
        act_1 = rearrange(act_1, " d e f->d f e ") 
        prompt_feature_1 = torch.matmul(act_1, cosin_1)   
        prompt_feature_1 = rearrange(prompt_feature_1, " d e f->d f e ")  

        act_t = act.permute(0, 2, 3, 1)    
        b_size, _, h_size, w_size = act.shape   
        act_t = rearrange(act_t, " a b c d->a (b c) d")    
        prompt_feature_1_yuan = torch.matmul(act_t, cosin_1) 
        prompt_feature_1_yuan = prompt_feature_1_yuan.reshape(b_size, h_size, w_size, -1)  
        prompt_feature_1_yuan = prompt_feature_1_yuan.permute(0, 3, 1, 2)  

        #Routing2
        p_out, a_out_2, out_2,again = self.conv_caps_2(prompt_2)
        correlation_2_2 = out_2   
        correlation_2_2 = rearrange(correlation_2_2, " d e f->d f e ")    
        cosin_2 = torch.matmul(correlation_2_1, correlation_2_2)/torch.matmul(torch.norm(correlation_2_1, dim=2,keepdim=True) , torch.norm(correlation_2_2, dim=1,keepdim=True))
        cosin_2 = self.sigmoid(cosin_2)  
        act_2 = act.transpose(dim0=2, dim1=3).flatten(2, 3)
        act_2 = rearrange(act_2, " d e f->d f e ") 
        prompt_feature_2 = torch.matmul(act_2, cosin_2)  
        prompt_feature_2 = rearrange(prompt_feature_2, " d e f->d f e ") 
        
        prompt_feature_2_yuan = torch.matmul(act_t, cosin_2)        
        prompt_feature_2_yuan = prompt_feature_2_yuan.reshape(b_size, h_size, w_size, -1)  
        prompt_feature_2_yuan = prompt_feature_2_yuan.permute(0, 3, 1, 2)

        #Routing3
        p_out, a_out, out_3,again = self.conv_caps_3(prompt_3)
        correlation_3_2 = out_3   
        correlation_3_2 = rearrange(correlation_3_2, " d e f->d f e ")   
        cosin_3 = torch.matmul(correlation_3_1, correlation_3_2)/torch.matmul(torch.norm(correlation_3_1, dim=2,keepdim=True) , torch.norm(correlation_3_2, dim=1,keepdim=True))
        cosin_3 = self.sigmoid(cosin_3)      
        act_3 = torch.flip(act.flatten(2, 3), dims=[-1])
        act_3 = rearrange(act_3, " d e f->d f e ") 
        prompt_feature_3 = torch.matmul(act_3, cosin_3)        
        prompt_feature_3 = rearrange(prompt_feature_3, " d e f->d f e ") 

        prompt_feature_3_yuan = torch.matmul(act_t, cosin_3)        
        prompt_feature_3_yuan = prompt_feature_3_yuan.reshape(b_size, h_size, w_size, -1)  
        prompt_feature_3_yuan = prompt_feature_3_yuan.permute(0, 3, 1, 2)

        #Routing4
        p_out, a_out, out_4,again = self.conv_caps_4(prompt_4)
        correlation_4_2 = out_4   
        correlation_4_2 = rearrange(correlation_4_2, " d e f->d f e ")   
        cosin_4 = torch.matmul(correlation_4_1, correlation_4_2)/torch.matmul(torch.norm(correlation_4_1, dim=2,keepdim=True) , torch.norm(correlation_4_2, dim=1,keepdim=True))
        cosin_4 = self.sigmoid(cosin_4)    
        act_4 = torch.flip(act.transpose(dim0=2, dim1=3).flatten(2, 3), dims=[-1])
        act_4 = rearrange(act_4, " d e f->d f e ") 
        prompt_feature_4 = torch.matmul(act_4, cosin_4) 
        prompt_feature_4 = rearrange(prompt_feature_4, " d e f->d f e ") 

        prompt_feature_4_yuan = torch.matmul(act_t, cosin_4)        
        prompt_feature_4_yuan = prompt_feature_4_yuan.reshape(b_size, h_size, w_size, -1)  
        prompt_feature_4_yuan = prompt_feature_4_yuan.permute(0, 3, 1, 2)

        y1 = torch.unsqueeze(prompt_feature_1,dim=1) 
        y2 = torch.unsqueeze(prompt_feature_2,dim=1) 
        y3 = torch.unsqueeze(prompt_feature_3,dim=1) 
        y4 = torch.unsqueeze(prompt_feature_4,dim=1) 

        y1 = y1.view((x.shape[0], 1, self.C, -1)).view(x.shape[0], self.C, self.feature_size, self.feature_size)
        y3 = y3.flip(dims=[-1]).view(x.shape[0], 1, self.C, -1).view(x.shape[0], self.C, self.feature_size, self.feature_size)
        y4 = y4.flip(dims=[-1]).view(x.shape[0], 1, self.C, -1)
        y2 = y2.view(x.shape[0], -1, self.feature_size, self.feature_size).transpose(dim0=2, dim1=3).contiguous().view(x.shape[0], 1, self.C, -1).view(x.shape[0], self.C, self.feature_size, self.feature_size)
        y4 = y4.view(x.shape[0], -1, self.feature_size, self.feature_size).transpose(dim0=2, dim1=3).contiguous().view(x.shape[0], 1, self.C, -1).view(x.shape[0], self.C, self.feature_size, self.feature_size)

        x_1 = self.conv_keep(x)
        Guide =torch.cat([x_1, mamba_y, prompt_feature_1_yuan, prompt_feature_2_yuan, prompt_feature_3_yuan, prompt_feature_4_yuan], dim=1)   
        Guide =self.conv(Guide) 
        Guide = self.bn(Guide)

        x_2 = self.conv_keep_2(x)
        Guide_2 =torch.cat([x_2, mamba_y, y1, y2, y3, y4], dim=1)  
        Guide_2 =self.conv_2(Guide_2) 
        Guide_2 = self.bn_2(Guide_2)

        Guide_3 =torch.cat([Guide, Guide_2], dim=1)
        Guide_3 =self.conv_3(Guide_3) 
        Guide_3 = self.bn_3(Guide_3)
        Guide = Guide_3
        return Guide 
    