import timm
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
from .capsule_mcr import *
from .Decoder import Decoder,decoder_module
from .fuse import fuse_8_4,fuse_16_8
from .fuse import PatchMerging as PatchM
from .Transformer import Transformer
from .Transformer import token_Transformer
import math
from lib_mamba.vmamba import *

'''
backbone: Vmamba-T
'''

class Network(nn.Module):
    def __init__(self, args,channels=128,A=256, B=32, C=32, D=32, FF=16,G=16,P=4,feature_size=22):
        super(Network, self).__init__()

        self.shared_encoder = VSSM(
        depths=[2, 2, 8, 2], dims=96, drop_path_rate=0.2, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v05_noz", 
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d"), 
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, posembed=False, imgsize=352, 
    )
        pretrained_dict = torch.load('/data1/MCRNet/vssm1_tiny_0230s_ckpt_epoch_264.pth')
        true_state_dict = pretrained_dict["model"]
        pretrained_dict = {key.replace("model.", ""): value for key, value in true_state_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.shared_encoder.state_dict()}
        self.shared_encoder.load_state_dict(pretrained_dict)
        self.up = nn.Sequential(
            nn.Conv2d(192, 96, kernel_size=1),nn.BatchNorm2d(96),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),nn.BatchNorm2d(96),nn.ReLU(True)
        )
        self.mlp32 = nn.Sequential(
                nn.Linear(args.encoder_dim[3], args.encoder_dim[2]),
                nn.GELU(),
                nn.Linear(args.encoder_dim[2], args.encoder_dim[2]),)
        self.mlp16 = nn.Sequential(
                nn.Linear(args.encoder_dim[2], args.dim),
                nn.GELU(),
                nn.Linear(args.dim, args.dim),)
        self.norm1 = nn.LayerNorm(args.dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(args.dim, args.embed_dim),
            nn.GELU(),
            nn.Linear(args.embed_dim, args.embed_dim),)
        self.fuse_32_16 = decoder_module(dim=args.embed_dim, token_dim=args.dim, img_size=args.img_size, ratio=16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.halfconcat = PatchM(dim=args.encoder_dim[1], feature_size = feature_size, out_channel=args.encoder_dim[2])
        self.concatFuse = nn.Sequential(
                nn.Linear(args.encoder_dim[2]*2, args.encoder_dim[2]),
                nn.GELU(),
                nn.Linear(args.encoder_dim[2], args.encoder_dim[2]),)
        self.mlp_16 = nn.Sequential(
                nn.Linear(args.encoder_dim[2], args.encoder_dim[1]),
                nn.GELU(),
                nn.Linear(args.encoder_dim[1], args.encoder_dim[1]),)
        self.mlp_8 = nn.Sequential(
                nn.Linear(args.encoder_dim[1], args.dim*2),
                nn.GELU(),
                nn.Linear(args.dim*2, args.dim*2),)
        self.fuse_16_8 = fuse_16_8(dim=192, token_dim=args.dim*2, img_size=args.img_size, ratio=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.norm_8 = nn.LayerNorm(args.dim*2)   
        self.mlp_8_2 = nn.Sequential(
            nn.Linear(args.dim*2, args.encoder_dim[1]),
            nn.GELU(),
            nn.Linear(args.encoder_dim[1], args.encoder_dim[1]),)
        self.mlp__8 = nn.Sequential(
                nn.Linear(args.encoder_dim[1], args.encoder_dim[0]),
                nn.GELU(),
                nn.Linear(args.encoder_dim[0], args.encoder_dim[0]),)
        self.mlp__4 = nn.Sequential(
                nn.Linear(args.encoder_dim[0], args.dim*4),
                nn.GELU(),
                nn.Linear(args.dim*4, args.dim*4),)
        self.fuse_8_4 = fuse_8_4(dim=96, token_dim=args.dim*4, img_size=args.img_size, ratio=4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.norm_4 = nn.LayerNorm(args.dim*4) 
        self.mlp_4_2 = nn.Sequential(
            nn.Linear(args.dim*4, args.encoder_dim[0]),
            nn.GELU(),
            nn.Linear(args.encoder_dim[0], args.encoder_dim[0]),)
        #capsule for part-whole relational analysis
        self.conv1 = nn.Conv2d(in_channels=args.encoder_dim[2], out_channels=A,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=A, eps=0.001,
                                 momentum=0.1, affine=True)
        self.relu1 = nn.ReLU(inplace=False)  
        self.semantics_1 = Semantics(A, B, C, 1, P, w=22, h=22, out_channel=args.encoder_dim[2],feature_size=feature_size)    
        self.conv = nn.Conv2d(args.encoder_dim[2]*2, args.encoder_dim[2], kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn = nn.BatchNorm2d(args.encoder_dim[2]) 

        self.transformer = Transformer(embed_dim=args.encoder_dim[2], depth=4, num_heads=6, mlp_ratio=3.)
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
        en_feats = self.shared_encoder(x)
        rgb_fea_1_8, rgb_fea_1_16, rgb_fea_1_32, x1 = en_feats
        rgb_fea_1_4 = torch.nn.functional.interpolate(self.up(rgb_fea_1_8), scale_factor=2, mode='bilinear', align_corners=False)
        rgb_fea_1_4 = rearrange(rgb_fea_1_4, " a b c d-> a (c d) b")   
        rgb_fea_1_8 = rearrange(rgb_fea_1_8, " a b c d-> a (c d) b")
        rgb_fea_1_16 = rearrange(rgb_fea_1_16, " a b c d-> a (c d) b")
        rgb_fea_1_32 = rearrange(rgb_fea_1_32, " a b c d-> a (c d) b")
        rgb_fea_1_32 = self.mlp32(rgb_fea_1_32)
        rgb_fea_1_16 = self.mlp16(rgb_fea_1_16)
        rgb_fea_1_16 = self.fuse_32_16(rgb_fea_1_32, rgb_fea_1_16)
        rgb_fea_1_16__ = self.mlp1(self.norm1(rgb_fea_1_16))
        rgb_fea_1_8_half = self.halfconcat(rgb_fea_1_8) #add 1/8
        rgb_fea_1_16 = self.concatFuse(torch.cat([rgb_fea_1_16__, rgb_fea_1_8_half], dim=2))
        rgb_fea_1_16_ = self.mlp_16(rgb_fea_1_16__)
        rgb_fea_1_8 = self.mlp_8(rgb_fea_1_8)
        rgb_fea_1_8 = self.fuse_16_8(rgb_fea_1_16_, rgb_fea_1_8)
        rgb_fea_1_8 = self.mlp_8_2(self.norm_8(rgb_fea_1_8))
        rgb_fea_1_8_ = self.mlp__8(rgb_fea_1_8)
        rgb_fea_1_4 = self.mlp__4(rgb_fea_1_4)
        rgb_fea_1_4 = self.fuse_8_4(rgb_fea_1_8_, rgb_fea_1_4)
        rgb_fea_1_4 = self.mlp_4_2(self.norm_4(rgb_fea_1_4))

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


