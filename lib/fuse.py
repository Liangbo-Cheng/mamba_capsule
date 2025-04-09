import torch
import torch.nn as nn

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, feature_size=44, out_channel=384):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, out_channel, bias=False)
        self.norm = norm_layer(4 * dim)
        self.feature_size = feature_size

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # H, W = 44, 44
        # H, W = 52, 52
        H = self.feature_size*2
        W = self.feature_size*2


        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class fuse_16_8(nn.Module):
    def __init__(self, dim=384, token_dim=64, img_size=224, ratio=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True):
        super(fuse_16_8, self).__init__()

        self.project = nn.Linear(token_dim, token_dim * kernel_size[0] * kernel_size[1])
        self.upsample = nn.Fold(output_size=(img_size // ratio,  img_size // ratio), kernel_size=kernel_size, stride=stride, padding=padding)
        self.fuse = fuse
        if self.fuse:
            self.concatFuse = nn.Sequential(
                nn.Linear(token_dim*2, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim),
            )
            #self.att = Token_performer(dim=token_dim, in_dim=token_dim, kernel_ratio=0.5)

            # project input feature to 64 dim
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
                nn.Linear(dim, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim),
            )

    def forward(self, dec_fea, enc_fea=None):
        
        B,_,C = dec_fea.shape
        if C == 192:
            # from 192 to 64
            dec_fea = self.mlp(self.norm(dec_fea))

        # [1] token upsampling by the proposed reverse T2T module
        dec_fea = self.project(dec_fea)
        # [B, H*W, token_dim*kernel_size*kernel_size]
        dec_fea = self.upsample(dec_fea.transpose(1, 2))
        B, C, _, _ = dec_fea.shape
        dec_fea = dec_fea.view(B, C, -1).transpose(1, 2)
        # [B, HW, C]

        if self.fuse:
            # [2] fuse encoder fea and decoder fea
            dec_fea = self.concatFuse(torch.cat([dec_fea, enc_fea], dim=2))
            #dec_fea = self.att(dec_fea)

        return dec_fea
    

class fuse_8_4(nn.Module):
    def __init__(self, dim=384, token_dim=64, img_size=224, ratio=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True):
        super(fuse_8_4, self).__init__()

        self.project = nn.Linear(token_dim, token_dim * kernel_size[0] * kernel_size[1])
        self.upsample = nn.Fold(output_size=(img_size // ratio,  img_size // ratio), kernel_size=kernel_size, stride=stride, padding=padding)
        self.fuse = fuse
        if self.fuse:
            self.concatFuse = nn.Sequential(
                nn.Linear(token_dim*2, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim),
            )
            #self.att = Token_performer(dim=token_dim, in_dim=token_dim, kernel_ratio=0.5)

            # project input feature to 64 dim
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
                nn.Linear(dim, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim),
            )
        self.dim = dim

    def forward(self, dec_fea, enc_fea=None):
        
        B,_,C = dec_fea.shape
        # if C == self.dim:
        if C == 96:

            # from 384 to 64
            dec_fea = self.mlp(self.norm(dec_fea))

        # [1] token upsampling by the proposed reverse T2T module
        dec_fea = self.project(dec_fea)
        # [B, H*W, token_dim*kernel_size*kernel_size]
        dec_fea = self.upsample(dec_fea.transpose(1, 2))
        B, C, _, _ = dec_fea.shape
        dec_fea = dec_fea.view(B, C, -1).transpose(1, 2)
        # [B, HW, C]

        if self.fuse:
            # [2] fuse encoder fea and decoder fea
            dec_fea = self.concatFuse(torch.cat([dec_fea, enc_fea], dim=2))
            #dec_fea = self.att(dec_fea)

        return dec_fea