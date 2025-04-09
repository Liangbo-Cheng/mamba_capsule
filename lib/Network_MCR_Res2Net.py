import timm
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
from .capsule_mcr import *
from .Decoder import Decoder,decoder_module
from .fuse import fuse_8_4,fuse_16_8,PatchMerging
from .Transformer import Transformer
from .Transformer import token_Transformer
import math
from lib.Res2Net_v1b import res2net50_v1b_26w_4s

'''
backbone: Res2Net
'''

class Network(nn.Module):
    def __init__(self, args,channels=128,A=256, B=32, C=32, D=32, FF=16,G=16,P=4,feature_size=22):
        super(Network, self).__init__()
        self.shared_encoder = res2net50_v1b_26w_4s()
        pretrained_dict = torch.load('/data1/MCRNet/res2net50_v1b_26w_4s-3cf99910.pth')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.shared_encoder.state_dict()}
        self.shared_encoder.load_state_dict(pretrained_dict)
        self.up = nn.Sequential(
            nn.Conv2d(channels//4, channels, kernel_size=1),nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),nn.BatchNorm2d(channels),nn.ReLU(True)
        )
        self.adjust_x4 = nn.Sequential(nn.Conv2d(2048, args.encoder_dim[3], kernel_size=1, padding=0),nn.BatchNorm2d(args.encoder_dim[3]),nn.ReLU(True),)
        self.adjust_x3 = nn.Sequential(nn.Conv2d(1024, args.encoder_dim[2], kernel_size=1, padding=0),nn.BatchNorm2d(args.encoder_dim[2]),nn.ReLU(True),)
        self.adjust_x2 = nn.Sequential(nn.Conv2d(512, args.encoder_dim[1], kernel_size=1, padding=0),nn.BatchNorm2d(args.encoder_dim[1]),nn.ReLU(True),)
        self.adjust_x1 = nn.Sequential(nn.Conv2d(256, args.encoder_dim[0], kernel_size=1, padding=0),nn.BatchNorm2d(args.encoder_dim[0]),nn.ReLU(True),)
       
        #capsule for part-whole relational analysis
        self.conv1 = nn.Conv2d(in_channels=args.encoder_dim[2], out_channels=A,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=A, eps=0.001,
                                 momentum=0.1, affine=True)
        self.relu1 = nn.ReLU(inplace=False)  
        self.semantics_1 = Semantics(A, B, C, 1, P, w=22, h=22, out_channel=args.encoder_dim[2],feature_size=feature_size)   
        self.conv = nn.Conv2d(args.encoder_dim[2]*2, args.encoder_dim[2], kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn = nn.BatchNorm2d(args.encoder_dim[2]) 

        self.transformer = Transformer(embed_dim=args.encoder_dim[2], depth=2, num_heads=6, mlp_ratio=3.)
        self.mlp_16_try1 = nn.Sequential(
                nn.Linear(args.encoder_dim[2], args.encoder_dim[1]),
                nn.GELU(),
                nn.Linear(args.encoder_dim[1], args.encoder_dim[1]),)
        self.mlp_8_try1 = nn.Sequential(
                nn.Linear(args.encoder_dim[1], args.dim*2),
                nn.GELU(),
                nn.Linear(args.dim*2, args.dim*2),)
        self.fuse_16_8_try1 = fuse_16_8(args.encoder_dim[1], token_dim=args.dim*2, img_size=args.img_size, ratio=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.norm_8_try1 = nn.LayerNorm(args.dim*2)   
        self.mlp_8_2_try1 = nn.Sequential(
            nn.Linear(args.dim*2, args.encoder_dim[1]),
            nn.GELU(),
            nn.Linear(args.encoder_dim[1], args.encoder_dim[1]),)
        self.transformer_1 = Transformer(embed_dim=args.encoder_dim[1], depth=2, num_heads=6, mlp_ratio=3.)
        self.token_trans = token_Transformer(embed_dim=args.encoder_dim[2], depth=4, num_heads=6, mlp_ratio=3.)
        self.decoder = Decoder(embed_dim=args.encoder_dim[2], token_dim=64, depth=2, img_size=args.img_size)

       
    def forward(self, x):
        image = x
        x = self.shared_encoder.conv1(x)
        x = self.shared_encoder.bn1(x)
        x = self.shared_encoder.relu(x)
        x = self.shared_encoder.maxpool(x)  
        x1 = self.shared_encoder.layer1(x)  
        x2 = self.shared_encoder.layer2(x1) 
        x3 = self.shared_encoder.layer3(x2) 
        x4 = self.shared_encoder.layer4(x3)
        rgb_fea_1_4 = rearrange(self.adjust_x1(x1), " a b c d-> a (c d) b")
        rgb_fea_1_8 = rearrange(self.adjust_x2(x2), " a b c d-> a (c d) b")
        rgb_fea_1_16 = rearrange(self.adjust_x3(x3), " a b c d-> a (c d) b")
        rgb_fea_1_32 = rearrange(self.adjust_x4(x4), " a b c d-> a (c d) b")

        #capsule for part-whole relational analysis
        rgb_fea_1_16_cap_ = rearrange(rgb_fea_1_16, " a (b c) d-> a d b c",b = int(math.sqrt(rgb_fea_1_16.shape[1])))   
        rgb_fea_1_16_cap = self.conv1(rgb_fea_1_16_cap_)   
        rgb_fea_1_16_cap = self.bn1(rgb_fea_1_16_cap)
        rgb_fea_1_16_cap = self.relu1(rgb_fea_1_16_cap)
        rgb_fea_1_16_cap = self.semantics_1(rgb_fea_1_16_cap)
        rgb_fea_1_16_addcap =self.conv(torch.cat([rgb_fea_1_16_cap_,rgb_fea_1_16_cap],dim=1)) 
        rgb_fea_1_16_addcap = self.bn(rgb_fea_1_16_addcap)
        rgb_fea_1_16_addcap = rearrange(rgb_fea_1_16_addcap, " a b c d-> a (c d) b")
        rgb_fea_1_16 = rgb_fea_1_16 + rgb_fea_1_16_addcap 
        
        rgb_fea_1_16 = self.transformer(rgb_fea_1_16) 
        rgb_fea_1_16_try = self.mlp_16_try1(rgb_fea_1_16)
        rgb_fea_1_8_try = self.mlp_8_try1(rgb_fea_1_8)
        rgb_fea_1_8_try = self.fuse_16_8_try1(rgb_fea_1_16_try, rgb_fea_1_8_try)
        rgb_fea_1_8 = self.mlp_8_2_try1(self.norm_8_try1(rgb_fea_1_8_try))
        rgb_fea_1_8 = self.transformer_1(rgb_fea_1_8)

        fea_1_16, saliency_tokens, contour_tokens, fea_16, saliency_tokens_tmp, contour_tokens_tmp, saliency_fea_1_16, contour_fea_1_16 = self.token_trans(rgb_fea_1_16, 10)
        outputs_saliency, outputs_contour, outputs_saliency_s, outputs_contour_s = self.decoder(fea_1_16, saliency_tokens, contour_tokens, fea_16, saliency_tokens_tmp, contour_tokens_tmp, saliency_fea_1_16, contour_fea_1_16, rgb_fea_1_8, rgb_fea_1_4,  10)

        return outputs_saliency, outputs_contour, outputs_saliency_s, outputs_contour_s


