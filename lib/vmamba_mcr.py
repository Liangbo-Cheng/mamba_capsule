import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from torchvision.models import VisionTransformer

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

try:
    from .csm_triton import CrossScanTriton, CrossMergeTriton, CrossScanTriton1b1, getCSM
    from .csms6s import CrossScan, CrossMerge
    from .csms6s import CrossScan_Ab_1direction, CrossMerge_Ab_1direction, CrossScan_Ab_2direction, CrossMerge_Ab_2direction
    from .csms6s import SelectiveScanMamba, SelectiveScanCore, SelectiveScanOflex
    from .csms6s import flops_selective_scan_fn, flops_selective_scan_ref, selective_scan_flop_jit
except:
    from csm_triton import CrossScanTriton, CrossMergeTriton, CrossScanTriton1b1, getCSM
    from csms6s import CrossScan, CrossMerge
    from csms6s import CrossScan_Ab_1direction, CrossMerge_Ab_1direction, CrossScan_Ab_2direction, CrossMerge_Ab_2direction
    from csms6s import SelectiveScanMamba, SelectiveScanCore, SelectiveScanOflex
    from csms6s import flops_selective_scan_fn, flops_selective_scan_ref, selective_scan_flop_jit

# =====================================================
# we have this class as linear and conv init differ from each other
# this function enable loading from both conv2d or linear
class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)



class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm, channel_first=False):
        super().__init__()
        self.dim = dim
        Linear = Linear2d if channel_first else nn.Linear
        self._patch_merging_pad = self._patch_merging_pad_channel_first if channel_first else self._patch_merging_pad_channel_last
        self.reduction = Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad_channel_last(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    @staticmethod
    def _patch_merging_pad_channel_first(x: torch.Tensor):
        H, W = x.shape[-2:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2]  # ... H/2 W/2
        x1 = x[..., 1::2, 0::2]  # ... H/2 W/2
        x2 = x[..., 0::2, 1::2]  # ... H/2 W/2
        x3 = x[..., 1::2, 1::2]  # ... H/2 W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x


# =====================================================
class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D


class SS2Dv0:
    # only used to run previous version
    def __initv0__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        # ======================
        dropout=0.0,
        # ======================
        seq=False,
        force_fp32=True,
        **kwargs,
    ):
        if "channel_first" in kwargs:
            assert not kwargs["channel_first"]
        act_layer = nn.SiLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        conv_bias = True
        d_conv = 3
        k_group = 4
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.forward = self.forwardv0 
        if seq:
            self.forward = partial(self.forwardv0, seq=True)
        if not force_fp32:
            self.forward = partial(self.forwardv0, force_fp32=False)

        # in proj ============================
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()
        self.conv2d = nn.Conv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            groups=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
            
        # A, D =======================================
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=k_group, merge=True) # (K * D)     

        # out proj =======================================
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    # only used to run previous version
    def forwardv0(self, x: torch.Tensor, SelectiveScan = SelectiveScanMamba, seq=False, force_fp32=True, **kwargs):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1) # (b, h, w, d)
        z = self.act(z)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x) # (b, d, h, w)
        x = self.act(x)
        
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, False)

        B, D, H, W = x.shape
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        L = H * W

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.contiguous() # (b, k, d_state, l)
        Cs = Cs.contiguous() # (b, k, d_state, l)
        
        As = -torch.exp(self.A_logs.float()) # (k * d, d_state)
        Ds = self.Ds.float() # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        
        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        if seq:
            out_y = []
            for i in range(4):
                yi = selective_scan(
                    xs.view(B, K, -1, L)[:, i], dts.view(B, K, -1, L)[:, i], 
                    As.view(K, -1, N)[i], Bs[:, i].unsqueeze(1), Cs[:, i].unsqueeze(1), Ds.view(K, -1)[i],
                    delta_bias=dt_projs_bias.view(K, -1)[i],
                    delta_softplus=True,
                ).view(B, -1, L)
                out_y.append(yi)
            out_y = torch.stack(out_y, dim=1)
        else:
            out_y = selective_scan(
                xs, dts, 
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        
        y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class SS2Dv2:
    def __initv2__(
        self,
        # basic dims ===========
        d_model=96,     
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,    
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_conv = d_conv
        self.channel_first = channel_first
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardv2

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.oact, forward_type = checkpostfix("oact", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        # softmax | sigmoid | dwconv | norm ===========================
        self.out_norm_shape = "v1"
        if forward_type[-len("none"):] == "none":
            forward_type = forward_type[:-len("none")]
            self.out_norm = nn.Identity()
        elif forward_type[-len("dwconv3"):] == "dwconv3":
            forward_type = forward_type[:-len("dwconv3")]
            self.out_norm = nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False)
        elif forward_type[-len("softmax"):] == "softmax":
            forward_type = forward_type[:-len("softmax")]
            class SoftmaxSpatial(nn.Softmax):
                def forward(self, x: torch.Tensor):
                    B, C, H, W = x.shape
                    return super().forward(x.view(B, C, -1)).view(B, C, H, W)
            self.out_norm = SoftmaxSpatial(dim=-1)
        elif forward_type[-len("sigmoid"):] == "sigmoid":
            forward_type = forward_type[:-len("sigmoid")]
            self.out_norm = nn.Sigmoid()
        elif channel_first:
            self.out_norm = LayerNorm2d(d_inner)
        else:
            self.out_norm_shape = "v0"
            self.out_norm = nn.LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba),
            v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanOflex, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v04=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v05=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            # ===============================
            v31d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, CrossScan=CrossScan_Ab_1direction, CrossMerge=CrossMerge_Ab_1direction,
            ),
            v32d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, CrossScan=CrossScan_Ab_2direction, CrossMerge=CrossMerge_Ab_2direction,
            ),
            v32dc=partial(self.forward_corev1, force_fp32=False, SelectiveScan=SelectiveScanOflex),
            # ===============================
            v051d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, CrossScan=getCSM(1)[0], CrossMerge=getCSM(1)[1],
            ),
            v052d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, CrossScan=getCSM(2)[0], CrossMerge=getCSM(2)[1],
            ),
            v052dc=partial(self.forward_corev1, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True),
            # ===============================
            v1=partial(self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanOflex),
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanCore),
            v3=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex),
            v4=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        k_group = 4

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        
        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(d_inner, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            # dt proj ============================
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
            del self.dt_projs
            
            # A, D =======================================
            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True) # (K * D)
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner))) 
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, d_inner)))

    # cascade2d
    def forward_corev1(
        self,
        x: torch.Tensor=None, 
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        delta_softplus = True,
        out_norm: torch.nn.Module=None,
        out_norm_shape="v0",
        channel_first=False,
        # ==============================
        to_dtype=True, # True: final out to dtype
        force_fp32=False, # True: input fp32
        # ==============================
        nrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
        backnrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
        ssoflex=True, # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
        # ==============================
        SelectiveScan=None,
        no_einsum=False, # replace einsum with linear or conv1d to raise throughput
        **kwargs,
    ):
        x_proj_weight = self.x_proj_weight
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        out_norm = getattr(self, "out_norm", None)
        out_norm_shape = getattr(self, "out_norm_shape", "v0")
        channel_first = self.channel_first

        # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        if nrows == 0:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1
            
        if backnrows == 0:
            if D % 4 == 0:
                backnrows = 4
            elif D % 3 == 0:
                backnrows = 3
            elif D % 2 == 0:
                backnrows = 2
            else:
                backnrows = 1

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)
        
        def scan_rowcol(
            x: torch.Tensor, 
            proj_weight: torch.Tensor, 
            proj_bias: torch.Tensor, 
            dt_weight: torch.Tensor, 
            dt_bias: torch.Tensor, # (2*c)
            _As: torch.Tensor, # As = -torch.exp(A_logs.to(torch.float))[:2,] # (2*c, d_state)
            _Ds: torch.Tensor,
            width = True,
        ):
            # x: (B, D, H, W)
            # proj_weight: (2 * D, (R+N+N))
            XB, XD, XH, XW = x.shape
            if width:
                _B, _D, _L = XB * XH, XD, XW
                xs = x.permute(0, 2, 1, 3).contiguous()
            else:
                _B, _D, _L = XB * XW, XD, XH
                xs = x.permute(0, 3, 1, 2).contiguous()
            xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2) # (B, H, 2, D, W)
            if no_einsum:
                x_dbl = F.conv1d(xs.view(_B, -1, _L), proj_weight.view(-1, _D, 1), bias=(proj_bias.view(-1) if proj_bias is not None else None), groups=2)
                dts, Bs, Cs = torch.split(x_dbl.view(_B, 2, -1, _L), [R, N, N], dim=2)
                dts = F.conv1d(dts.contiguous().view(_B, -1, _L), dt_weight.view(2 * _D, -1, 1), groups=2)
            else:
                x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, proj_weight)
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, 2, -1, 1)
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_weight)

            xs = xs.view(_B, -1, _L)
            dts = dts.contiguous().view(_B, -1, _L)
            As = _As.view(-1, N).to(torch.float)
            Bs = Bs.contiguous().view(_B, 2, N, _L)
            Cs = Cs.contiguous().view(_B, 2, N, _L)
            Ds = _Ds.view(-1)
            delta_bias = dt_bias.view(-1).to(torch.float)

            if force_fp32:
                xs = xs.to(torch.float)
            dts = dts.to(xs.dtype)
            Bs = Bs.to(xs.dtype)
            Cs = Cs.to(xs.dtype)

            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            ).view(_B, 2, -1, _L)
            return ys
        
        As = -torch.exp(A_logs.to(torch.float)).view(4, -1, N)
        y_row = scan_rowcol(
            x,
            proj_weight = x_proj_weight.view(4, -1, D)[:2].contiguous(), 
            proj_bias = (x_proj_bias.view(4, -1)[:2].contiguous() if x_proj_bias is not None else None),
            dt_weight = dt_projs_weight.view(4, D, -1)[:2].contiguous(),
            dt_bias = (dt_projs_bias.view(4, -1)[:2].contiguous() if dt_projs_bias is not None else None),
            _As = As[:2].contiguous().view(-1, N),
            _Ds = Ds.view(4, -1)[:2].contiguous().view(-1),
            width=True,
        ).view(B, H, 2, -1, W).sum(dim=2).permute(0, 2, 1, 3)
        y_col = scan_rowcol(
            y_row,
            proj_weight = x_proj_weight.view(4, -1, D)[2:].contiguous().to(y_row.dtype), 
            proj_bias = (x_proj_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if x_proj_bias is not None else None),
            dt_weight = dt_projs_weight.view(4, D, -1)[2:].contiguous().to(y_row.dtype),
            dt_bias = (dt_projs_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if dt_projs_bias is not None else None),
            _As = As[2:].contiguous().view(-1, N),
            _Ds = Ds.view(4, -1)[2:].contiguous().view(-1),
            width=False,
        ).view(B, W, 2, -1, H).sum(dim=2).permute(0, 2, 3, 1)
        y = y_col

        if channel_first:
            y = y.view(B, -1, H, W)
            if out_norm_shape in ["v1"]:
                y = out_norm(y)
            else:
                y = out_norm(y.permute(0, 2, 3, 1))
                y = y.permute(0, 3, 1, 2)
            return (y.to(x.dtype) if to_dtype else y)

        if out_norm_shape in ["v1"]: # (B, C, H, W)
            y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1) # (B, H, W, C)
        else: # (B, L, C)
            y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
            y = out_norm(y).view(B, H, W, -1)

        return (y.to(x.dtype) if to_dtype else y)

    # Note: we did not use csm_triton in and before vssm1_0230, we used pytorch version !
    # Note: we did not use no_einsum in and before vssm1_0230, we used einsum version !    
    def forward_corev2(
        self,
        x: torch.Tensor=None, 
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        delta_softplus = True,
        out_norm: torch.nn.Module=None,
        out_norm_shape="v0",
        channel_first=False,
        # ==============================
        to_dtype=True, # True: final out to dtype
        force_fp32=False, # True: input fp32
        # ==============================
        nrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
        backnrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
        ssoflex=True, # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
        # ==============================
        SelectiveScan=None,
        CrossScan=CrossScan,
        CrossMerge=CrossMerge,
        no_einsum=False, # replace einsum with linear or conv1d to raise throughput
        **kwargs,
    ):
        x_proj_weight = self.x_proj_weight     
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        out_norm = getattr(self, "out_norm", None)
        out_norm_shape = getattr(self, "out_norm_shape", "v0")
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

        B, D, H, W = x.shape     
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        if nrows == 0:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1
            
        if backnrows == 0:
            if D % 4 == 0:
                backnrows = 4
            elif D % 3 == 0:
                backnrows = 3
            elif D % 2 == 0:
                backnrows = 2
            else:
                backnrows = 1

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)
       
        xs = CrossScan.apply(x)  
        if no_einsum:
            x_dbl = F.conv1d(xs.view(B, -1, L), x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
            dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * D, -1, 1), groups=K)
        else:
            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)    
            if x_proj_bias is not None:
                x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)      
            dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)  
       
        xs = xs.view(B, -1, L)    
        dts = dts.contiguous().view(B, -1, L)                    
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().view(B, K, N, L)                     
        Cs = Cs.contiguous().view(B, K, N, L)                     
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)
        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        )    
        hidden_state = ys[1]
        hidden_state= rearrange(hidden_state, " a (b e) c d-> a b e c d",b =4)
        ys = ys[0].view(B, K, -1, H, W)       

      
        y: torch.Tensor = CrossMerge.apply(ys)     

        if getattr(self, "__DEBUG__", False):
            setattr(self, "__data__", dict(
                A_logs=A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                us=xs, dts=dts, delta_bias=delta_bias,
                ys=ys, y=y,
            ))
        #LN层
        if channel_first:
            y = y.view(B, -1, H, W)
            if out_norm_shape in ["v1"]:
                y = out_norm(y)
            else:
                y = out_norm(y.permute(0, 2, 3, 1))
                y = y.permute(0, 3, 1, 2)
            return (y.to(x.dtype) if to_dtype else y) ,hidden_state

        if out_norm_shape in ["v1"]: # (B, C, H, W)
            y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1) # (B, H, W, C)
        else: # (B, L, C)
            y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
            y = out_norm(y).view(B, H, W, -1)

        return (y.to(x.dtype) if to_dtype else y), hidden_state

    def forwardv2(self, x: torch.Tensor, **kwargs):
        with_dconv = (self.d_conv > 1)
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1)) # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if with_dconv:
            x = self.conv2d(x) 
        x = self.act(x)       
        y, hidden_state = self.forward_core(x)    
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out, hidden_state


class SS2Dv3:
    def __initxv__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================              
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_conv = d_conv
        self.channel_first = channel_first
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.d_inner = d_inner
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardxv

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)

        # softmax | sigmoid | dwconv | norm ===========================
        self.out_norm_shape = "v1"
        if forward_type[-len("none"):] == "none":
            forward_type = forward_type[:-len("none")]
            self.out_norm = nn.Identity()
        elif forward_type[-len("dwconv3"):] == "dwconv3":
            forward_type = forward_type[:-len("dwconv3")]
            self.out_norm = nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False)
        elif forward_type[-len("softmax"):] == "softmax":
            forward_type = forward_type[:-len("softmax")]
            class SoftmaxSpatial(nn.Softmax):
                def forward(self, x: torch.Tensor):
                    B, C, H, W = x.shape
                    return super().forward(x.view(B, C, -1)).view(B, C, H, W)
            self.out_norm = SoftmaxSpatial(dim=-1)
        elif forward_type[-len("sigmoid"):] == "sigmoid":
            forward_type = forward_type[:-len("sigmoid")]
            self.out_norm = nn.Sigmoid()
        elif channel_first:
            self.out_norm = LayerNorm2d(d_inner)
        else:
            self.out_norm_shape = "v0"
            self.out_norm = nn.LayerNorm(d_inner)

        k_group = 4
        # in proj =======================================
        self.out_act: nn.Module = nn.Identity()
        # 0309 -> 0319 needs to be rerun...
        if False:
            # change Conv2d to Linear2d Next
            if forward_type.startswith("xv1"):
                self.in_proj = nn.Conv2d(d_model, d_inner + dt_rank + 8 * d_state, 1, bias=bias, **factory_kwargs)

            if forward_type.startswith("xv2"):
                self.in_proj = nn.Conv2d(d_model, d_inner + d_inner + 8 * d_state, 1, bias=bias, **factory_kwargs)
                self.forward = partial(self.forwardxv, mode="xv2")
                del self.dt_projs_weight

            if forward_type.startswith("xv3"):
                self.forward = partial(self.forwardxv, mode="xv3")
                self.in_proj = nn.Conv2d(d_model, d_inner + 4 * dt_rank + 8 * d_state, 1, bias=bias, **factory_kwargs)

            if forward_type.startswith("xv4"):
                self.forward = partial(self.forwardxv, mode="xv3")
                self.in_proj = nn.Conv2d(d_model, d_inner + 4 * dt_rank + 8 * d_state, 1, bias=bias, **factory_kwargs)
                self.out_act = nn.GELU()

            if forward_type.startswith("xv5"):
                self.in_proj = nn.Conv2d(d_model, d_inner + d_inner + 8 * d_state, 1, bias=bias, **factory_kwargs)
                self.forward = partial(self.forwardxv, mode="xv2")
                del self.dt_projs_weight
                self.out_act = nn.GELU()

            if forward_type.startswith("xv6"):
                self.forward = partial(self.forwardxv, mode="xv1")
                self.in_proj = nn.Conv2d(d_model, d_inner + dt_rank + 8 * d_state, 1, bias=bias, **factory_kwargs)
                self.out_act = nn.GELU()

            # to see if Linear2d and nn.Conv2d differ, as they will be inited differ
            if forward_type.startswith("xv61"):
                self.forward = partial(self.forwardxv, mode="xv1")
                self.in_proj = Linear2d(d_model, d_inner + dt_rank + 8 * d_state, bias=bias, **factory_kwargs)
                self.out_act = nn.GELU()
            
            if forward_type.startswith("xv7"):
                self.forward = partial(self.forwardxv, mode="xv1", omul=True)
                self.in_proj = Linear2d(d_model, d_inner + dt_rank + 8 * d_state, bias=bias, **factory_kwargs)
                self.out_act = nn.GELU()
            
        if True:
            self.ocov2, forward_type = checkpostfix("ocov2", forward_type)
            self.ocov, forward_type = checkpostfix("ocov", forward_type)
            self.omul, forward_type = checkpostfix("mul", forward_type)
            if self.ocov2:
                self.f_ocov2 = nn.Identity()
            if self.ocov:
                self.f_ocov = nn.Identity()
            if self.omul:
                self.f_omul = nn.Identity()
            oact, forward_type = checkpostfix("act", forward_type)
            self.out_act = nn.GELU() if oact else nn.Identity()

            if forward_type.startswith("xv1a"):
                self.forward = partial(self.forwardxv, mode="xv1a")
                d_inner_all = d_inner + dt_rank + 8 * d_state
            elif forward_type.startswith("xv2a"):
                self.forward = partial(self.forwardxv, mode="xv2a")
                d_inner_all = d_inner + d_inner + 8 * d_state
            elif forward_type.startswith("xv3a"):
                self.forward = partial(self.forwardxv, mode="xv3a")
                d_inner_all = d_inner + 4 * dt_rank + 8 * d_state
            
            self.in_proj = Linear2d(d_model, d_inner_all, bias=bias, **factory_kwargs)

        # conv =======================================
        if d_conv > 1:
            cact, forward_type = checkpostfix("ca", forward_type)
            self.act: nn.Module = act_layer() if cact else nn.Identity()
                
            if self.ocov:
                self.oconv2d = nn.Conv2d(
                    in_channels=d_inner,
                    out_channels=d_inner,
                    groups=d_inner,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                    **factory_kwargs,
                )
            elif self.ocov2:
                self.conv2d = nn.Conv2d(
                    in_channels=d_inner_all,
                    out_channels=d_inner_all,
                    groups=d_inner_all,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                    **factory_kwargs,
                )
            else:
                self.conv2d = nn.Conv2d(
                    in_channels=d_model,
                    out_channels=d_model,
                    groups=d_model,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                    **factory_kwargs,
                )

        # out proj =======================================
        self.out_proj = Linear(d_inner, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            # dt proj ============================
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
            del self.dt_projs
            
            # A, D =======================================
            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True) # (K * D)
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner))) 
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, d_inner)))

        if forward_type.startswith("xv2"):
            del self.dt_projs_weight

    def forwardxv(self, x: torch.Tensor, mode="xv1a", omul=False, **kwargs):
        B, C, H, W = x.shape
        if not self.channel_first:
            B, H, W, C = x.shape
        L = H * W
        K = 4
        dt_projs_weight = getattr(self, "dt_projs_weight", None)
        A_logs = self.A_logs
        dt_projs_bias = self.dt_projs_bias
        force_fp32 = False
        delta_softplus = True
        out_norm_shape = getattr(self, "out_norm_shape", "v0")
        out_norm = self.out_norm
        to_dtype = True
        Ds = self.Ds

        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        def selective_scan(u, delta, A, B, C, D, delta_bias, delta_softplus):
            return SelectiveScanOflex.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, 1, True)

        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()

        if (self.d_conv > 1) and (not self.ocov) and (not self.ocov2):
            x = self.conv2d(x) # (b, d, h, w)
            x = self.act(x)

        x = self.in_proj(x)
        if (self.d_conv > 1) and (self.ocov2):
            x = self.conv2d(x) # (b, d, h, w)

        if mode in ["xv1", "xv2", "xv3", "xv7"]:
            print(f"ERROR: MODE {mode} will be deleted in the future, use {mode}a instead.")

        if mode in ["xv1"]:
            _us, dts, Bs, Cs = x.split([self.d_inner, self.dt_rank, 4 * self.d_state, 4 * self.d_state], dim=1)
            us = CrossScanTriton.apply(_us.contiguous()).view(B, -1, L)
            dts = CrossScanTriton.apply(dts.contiguous()).view(B, -1, L)
            dts = F.conv1d(dts, dt_projs_weight.view(K * self.d_inner, self.dt_rank, 1), None, groups=K).contiguous().view(B, -1, L)
        elif mode in ["xv2"]:
            _us, dts, Bs, Cs = x.split([self.d_inner, self.d_inner, 4 * self.d_state, 4 * self.d_state], dim=1)
            us = CrossScanTriton.apply(_us.contiguous()).view(B, -1, L)
            dts = CrossScanTriton.apply(dts).contiguous().view(B, -1, L)
        elif mode in ["xv3"]:
            _us, dts, Bs, Cs = x.split([self.d_inner, 4 * self.dt_rank, 4 * self.d_state, 4 * self.d_state], dim=1)
            us = CrossScanTriton.apply(_us.contiguous()).view(B, -1, L)
            dts = CrossScanTriton1b1.apply(dts.contiguous().view(B, K, -1, H, W))
            dts = F.conv1d(dts.view(B, -1, L), dt_projs_weight.view(K * self.d_inner, self.dt_rank, 1), None, groups=K).contiguous().view(B, -1, L)
        else:
            ...

        if mode in ["xv1a"]:
            us, dts, Bs, Cs = x.split([self.d_inner, self.dt_rank, 4 * self.d_state, 4 * self.d_state], dim=1)
            _us = us
            us = CrossScanTriton.apply(us.contiguous()).view(B, 4, -1, L)
            dts = CrossScanTriton.apply(dts.contiguous()).view(B, 4, -1, L)
            Bs = CrossScanTriton1b1.apply(Bs.view(B, 4, -1, H, W).contiguous()).view(B, 4, -1, L)
            Cs = CrossScanTriton1b1.apply(Cs.view(B, 4, -1, H, W).contiguous()).view(B, 4, -1, L)
            dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * self.d_inner, self.dt_rank, 1), None, groups=K)
            us, dts = us.contiguous().view(B, -1, L), dts
            _us = us.view(B, K, -1, H, W)[:, 0, :, :, :]
        elif mode in ["xv2a"]:
            us, dts, Bs, Cs = x.split([self.d_inner, self.d_inner, 4 * self.d_state, 4 * self.d_state], dim=1)
            _us = us
            us = CrossScanTriton.apply(us.contiguous()).view(B, 4, -1, L)
            dts = CrossScanTriton.apply(dts.contiguous()).view(B, 4, -1, L)
            Bs = CrossScanTriton1b1.apply(Bs.view(B, 4, -1, H, W).contiguous()).view(B, 4, -1, L)
            Cs = CrossScanTriton1b1.apply(Cs.view(B, 4, -1, H, W).contiguous()).view(B, 4, -1, L)
            us, dts = us.contiguous().view(B, -1, L), dts.contiguous().view(B, -1, L)
        elif mode in ["xv3a"]:
            # us, dtBCs = x.split([self.d_inner, 4 * self.dt_rank + 4 * self.d_state + 4 * self.d_state], dim=1)
            # _us = us
            # us = CrossScanTriton.apply(us.contiguous()).view(B, 4, -1, L)
            # dtBCs = CrossScanTriton1b1.apply(dtBCs.view(B, 4, -1, H, W).contiguous()).view(B, 4, -1, L)
            # dts, Bs, Cs = dtBCs.split([self.dt_rank, self.d_state, self.d_state], dim=2)
            # dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * self.d_inner, self.dt_rank, 1), None, groups=K)
            # us, dts = us.contiguous().view(B, -1, L), dts
            
            us, dts, Bs, Cs = x.split([self.d_inner, 4 * self.dt_rank, 4 * self.d_state, 4 * self.d_state], dim=1)
            _us = us
            us = CrossScanTriton.apply(us.contiguous()).view(B, 4, -1, L)
            dts = CrossScanTriton1b1.apply(dts.view(B, 4, -1, H, W).contiguous()).view(B, 4, -1, L)
            Bs = CrossScanTriton1b1.apply(Bs.view(B, 4, -1, H, W).contiguous()).view(B, 4, -1, L)
            Cs = CrossScanTriton1b1.apply(Cs.view(B, 4, -1, H, W).contiguous()).view(B, 4, -1, L)
            dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * self.d_inner, self.dt_rank, 1), None, groups=K)
            us, dts = us.contiguous().view(B, -1, L), dts
        else: 
            ...

        Bs, Cs = Bs.view(B, K, -1, L).contiguous(), Cs.view(B, K, -1, L).contiguous()
    
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float) # (K * c)

        if force_fp32:
            us, dts, Bs, Cs = to_fp32(us, dts, Bs, Cs)

        ys: torch.Tensor = selective_scan(
            us, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, H, W)
            
        y: torch.Tensor = CrossMergeTriton.apply(ys)
        y = y.view(B, -1, H, W)

        if getattr(self, "__DEBUG__", False):
            setattr(self, "__data__", dict(
                A_logs=A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                us=us, dts=dts, delta_bias=delta_bias,
                ys=ys, y=y,
            ))

        # originally:
        # y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
        # y = out_norm(y).view(B, H, W, -1)

        if (not self.channel_first) or (out_norm_shape in ["v0"]):
            y = out_norm(y.permute(0, 2, 3, 1))
            if self.channel_first:
                y = y.permute(0, 3, 1, 2)
        else:
            y = out_norm(y)

        y = (y.to(x.dtype) if to_dtype else y)
        y = self.out_act(y)
        if self.omul:
            y = y * (_us.permute(0, 2, 3, 1) if not self.channel_first else _us)
        elif self.ocov:
            y = y + self.act(self.oconv2d(_us))

        out = self.dropout(self.out_proj(y))
        return out


class SS2D(nn.Module, mamba_init, SS2Dv0, SS2Dv2, SS2Dv3):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,
    ):
        super().__init__()
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first,
        )
        if forward_type in ["v0", "v0seq"]:
            self.__initv0__(seq=("seq" in forward_type), **kwargs)
            return
        elif forward_type.startswith("xv"):
            self.__initxv__(**kwargs)
            return
        else:
            self.__initv2__(**kwargs)
            return


# =====================================================
class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        # =============================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp=False,
        # =============================
        use_checkpoint: bool = False,
        post_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim, 
                d_state=ssm_d_state, 
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
            )
        
        self.drop_path = DropPath(drop_path)
        
        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, input: torch.Tensor):
        if isinstance(input, tuple):
            x, b = input
        else:
            x = input
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                a, b = self.op(self.norm(x))
                x = x + self.drop_path(a)
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x))) # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        return x, b

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


class VSSM(nn.Module):
    def __init__(
        self, 
        patch_size=4, 
        in_chans=3, 
        num_classes=1000, 
        depths=[2], 
        dims=[1088], 
        # =========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",        
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # =========================
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        # =========================
        drop_path_rate=0.1, 
        patch_norm=True, 
        norm_layer="LN", # "BN", "LN2D"
        downsample_version: str = "v2", # "v1", "v2", "v3"
        patchembed_version: str = "v1", # "v1", "v2"
        use_checkpoint=False,  
        **kwargs,
    ):
        super().__init__()
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)

        _make_patch_embed = dict(
            v1=self._make_patch_embed, 
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer, channel_first=self.channel_first)

        _make_downsample = dict(
            v1=PatchMerging2D, 
            v2=self._make_downsample, 
            v3=self._make_downsample_v3, 
            # vno=self._make_downsample_no, 
            none=(lambda *_, **_k: None),
        ).get(downsample_version, None)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer], 
                self.dims[i_layer + 1], 
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim = self.dims[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
            ))

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer], 
                self.dims[i_layer + 1], 
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim = self.dims[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
            ))

        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)
        # self.plug_44 = nn.Linear(192, 96)
        # self.plug_44_2 = nn.Linear(768, 192)
        # # self.plug_88_2 = nn.Linear(768, 96)
        # self.plug_88 = nn.Linear(192, 96)
        
        
        # self.downsample = PatchMerging((88,88), dim=96, norm_layer=norm_layer)
        # self.upsample = upsample_module()

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # used in building optimizer
    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {}

    # used in building optimizer
    # @torch.jit.ignore
    # def no_weight_decay_keywords(self):
    #     return {}

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )
    
    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=22, stride=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )
    

    # def _make_downsample_no(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
    #     # if channel first, then Norm and Output are both channel_first
    #     return nn.Sequential(
    #         (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
    #         nn.Conv2d(dim, out_dim, kernel_size=1, stride=1),
    #         (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
    #         norm_layer(out_dim),
    #     )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
        dim=96, 
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        channel_first=False,
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        gmlp=False,
        **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
            ))
        
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks,),
            downsample=downsample,
        ))
    

    def forward(self, x: torch.Tensor):   
        if x.shape[1] == 22:
            for layer in self.layers:
                x = layer(x)
        return x

  

    def flops(self, shape=(3, 224, 224)):
        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScanMamba": partial(selective_scan_flop_jit, flops_fn=flops_selective_scan_fn),
            "prim::PythonOp.SelectiveScanOflex": partial(selective_scan_flop_jit, flops_fn=flops_selective_scan_fn),
            "prim::PythonOp.SelectiveScanCore": partial(selective_scan_flop_jit, flops_fn=flops_selective_scan_fn),
            "prim::PythonOp.SelectiveScanNRow": partial(selective_scan_flop_jit, flops_fn=flops_selective_scan_fn),
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"

    # used to load ckpt from previous training code
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):

        def check_name(src, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    return True
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        return True
            return False

        def change_name(src, dst, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    state_dict[prefix + dst] = state_dict[prefix + src]
                    state_dict.pop(prefix + src)
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        new_k = prefix + dst + k[len(key):]
                        state_dict[new_k] = state_dict[k]
                        state_dict.pop(k)

        change_name("patch_embed.proj", "patch_embed.0")
        change_name("patch_embed.norm", "patch_embed.2")
        for i in range(100):
            for j in range(100):
                change_name(f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm")
                change_name(f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op")
        change_name("norm", "classifier.norm")
        change_name("head", "classifier.head")

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


# compatible with openmmlab
class Backbone_VSSM(VSSM):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer="ln", **kwargs):
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)        
        
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return
        
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)        
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x) # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        if len(self.out_indices) == 0:
            return x
        
        return outs
    

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
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

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class upsample_module(nn.Module):
    def __init__(self, dim=768, token_dim=64, img_size=352, ratio=4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=False):
        super(upsample_module, self).__init__()

        self.project = nn.Linear(token_dim, token_dim * kernel_size[0] * kernel_size[1])
        self.project_ = nn.Linear(64, 96)
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
        if C == 768:
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
        dec_fea = self.project_(dec_fea)
        return dec_fea
    

class VSSM_cap(nn.Module):
    def __init__(
        self, 
        patch_size=4, 
        in_chans=3, 
        num_classes=1000, 
        depths=[2], 
        dims=[48],    
        # =========================
        ssm_d_state=8,      
        ssm_ratio=2.0,       
        ssm_dt_rank="auto",
        ssm_act_layer="silu",        
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # =========================
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        # =========================
        drop_path_rate=0.1, 
        patch_norm=True, 
        norm_layer="LN", # "BN", "LN2D"
        downsample_version: str = "v2", # "v1", "v2", "v3"
        patchembed_version: str = "v1", # "v1", "v2"
        use_checkpoint=False,  
        **kwargs,
    ):
        super().__init__()
        self.depths = depths
        self.dims =dims
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)

        _make_patch_embed = dict(
            v1=self._make_patch_embed, 
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer, channel_first=self.channel_first)

        _make_downsample = dict(
            v1=PatchMerging2D, 
            v2=self._make_downsample, 
            v3=self._make_downsample_v3, 
            # vno=self._make_downsample_no, 
            none=(lambda *_, **_k: None),
        ).get(downsample_version, None)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer], 
                self.dims[i_layer + 1], 
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim = self.dims[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
            ))

        self.layers_1 = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer], 
                self.dims[i_layer + 1], 
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers_1.append(self._make_layer(
                dim = self.dims[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
            ))

        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)
        self.plug_44 = nn.Linear(192, 96)
        self.plug_44_2 = nn.Linear(768, 192)
        # self.plug_88_2 = nn.Linear(768, 96)
        self.plug_88 = nn.Linear(192, 96)
        
        
        self.downsample = PatchMerging((88,88), dim=96, norm_layer=norm_layer)
        self.upsample = upsample_module()

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # used in building optimizer
    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {}

    # used in building optimizer
    # @torch.jit.ignore
    # def no_weight_decay_keywords(self):
    #     return {}

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )
    
    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=1, stride=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )
    

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
        dim=96, 
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        channel_first=False,
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        gmlp=False,
        **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
            ))
        
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks,),
            downsample=downsample,
        ))
    

    def forward(self, x: torch.Tensor):  
        
        for layer in self.layers:
            x, hidden_state = layer(x)
        return x, hidden_state

     
    def flops(self, shape=(3, 224, 224)):
        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScanMamba": partial(selective_scan_flop_jit, flops_fn=flops_selective_scan_fn),
            "prim::PythonOp.SelectiveScanOflex": partial(selective_scan_flop_jit, flops_fn=flops_selective_scan_fn),
            "prim::PythonOp.SelectiveScanCore": partial(selective_scan_flop_jit, flops_fn=flops_selective_scan_fn),
            "prim::PythonOp.SelectiveScanNRow": partial(selective_scan_flop_jit, flops_fn=flops_selective_scan_fn),
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"

    # used to load ckpt from previous training code
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):

        def check_name(src, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    return True
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        return True
            return False

        def change_name(src, dst, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    state_dict[prefix + dst] = state_dict[prefix + src]
                    state_dict.pop(prefix + src)
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        new_k = prefix + dst + k[len(key):]
                        state_dict[new_k] = state_dict[k]
                        state_dict.pop(k)

        change_name("patch_embed.proj", "patch_embed.0")
        change_name("patch_embed.norm", "patch_embed.2")
        for i in range(100):
            for j in range(100):
                change_name(f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm")
                change_name(f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op")
        change_name("norm", "classifier.norm")
        change_name("head", "classifier.head")

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
