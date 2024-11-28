## pytorch_tps is obtained from https://github.com/cheind/py-thin-plate-spline/blob/master/thinplate/pytorch.py
## the architecture of IDNet is based on the code from https://github.com/BingyaoHuang/CompenNeSt-plusplus/blob/master/src/python/Models.py

# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from torch.nn import GroupNorm
import torch.nn.functional as F
import pytorch_tps
import copy
from attention.non_local_coordinate import NonLocal
from mmcv.ops import DeformConv2dPack

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class skip(nn.Module):
    def __init__(self, kernel=1,stride=1, padding=0, in_channel=32, out_channel=64):
        super().__init__()
        self.proj1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=stride, padding=1)
        self.proj2 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=stride, padding=0)
        self.proj3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)        
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.act(self.proj1(x))
        x2 = self.act(self.proj2(x))
        x = self.proj3(x1 + x2)
        return x

class DeformableILM(nn.Module):
    def __init__(self, in_chans=3, height=256, width=256, batch=10):
        super(DeformableILM, self).__init__()
        self.deform_conv3 = DeformConv2dPack(in_channels=in_chans, out_channels=in_chans, kernel_size=3, padding=1)
        self.deform_conv5 = DeformConv2dPack(in_channels=in_chans, out_channels=in_chans, kernel_size=5, padding=2)
        self.deform_conv7 = DeformConv2dPack(in_channels=in_chans, out_channels=in_chans, kernel_size=7, padding=3)
        self.act = nn.GELU()

    def forward(self, x):
        x3 = self.act(self.deform_conv3(x))
        x5 = self.act(self.deform_conv5(x))
        x7 = self.act(self.deform_conv7(x))
        x = x3+x5+x7+x
        return x

class ILM(nn.Module):
    def __init__(self, patch_size=16, stride=2, padding=0, 
                 in_chans=3, embed_dim=128, norm_layer=None):
        super().__init__()
        self.proj7 = nn.Conv2d(in_chans, embed_dim, kernel_size=7, stride=stride, padding=3)
        self.proj5 = nn.Conv2d(in_chans, embed_dim, kernel_size=5, stride=stride, padding=2)
        self.proj3 = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=stride, padding=1)
        self.res = nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=stride, padding=0) #change input dimension to output dimension
        self.act = nn.ReLU()

    def forward(self, x):
        x7 = self.act(self.proj7(x))
        x5 = self.act(self.proj5(x))
        x3 = self.act(self.proj3(x))
        x = self.act(self.res(x))
        x = x7+x5+x3+x
        return x

class NLM(nn.Module):
    def __init__(self, patch_size=16, stride=1, padding=0, 
                 in_chans=128, embed_dim=128, norm_layer=None):
        super().__init__()
        self.proj7 = nn.Conv2d(in_chans, in_chans, kernel_size=7, stride=stride, padding=3)
        self.proj5 = nn.Conv2d(in_chans, in_chans, kernel_size=5, stride=stride, padding=2)
        self.proj3 = nn.Conv2d(in_chans, in_chans, kernel_size=3, stride=stride, padding=1)
        self.NL = NonLocal(in_chans)
        self.act = nn.ReLU()

    def forward(self, x):
        x7 = self.act(self.proj7(x))
        x5 = self.act(self.proj5(x))
        x3 = self.act(self.proj3(x))
        xn = self.act(self.NL(x))
        x = x7+x5+x3+xn+x
        return x

class UpSample(nn.Module):
    def __init__(self, patch_size=2, stride=2, padding=0, 
                 in_chans=320, embed_dim=128, norm_layer=None):
        super().__init__()
        self.proj = nn.ConvTranspose2d(in_chans, embed_dim, kernel_size=patch_size, 
                              stride=stride, padding=padding)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.act(x)
        return x

class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Pooling(nn.Module):
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PoolResblock(nn.Module):
    """
    Implementation of Pool operation based resblock.
    refer to https://arxiv.org/abs/2111.11418
    """
    def __init__(self, dim, pool_size=3, mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop=0., drop_path=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

def basic_blocks(dim, index, layers, 
                 pool_size=3, mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop_rate=.0, drop_path_rate=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(PoolResblock(
            dim, pool_size=pool_size, mlp_ratio=mlp_ratio, 
            act_layer=act_layer, norm_layer=norm_layer, 
            drop=drop_rate, drop_path=block_dpr, 
            use_layer_scale=use_layer_scale, 
            layer_scale_init_value=layer_scale_init_value, 
            ))
    blocks = nn.Sequential(*blocks)

    return blocks


class PCN(nn.Module):
    '''
    Implementation of PCN network.
    '''
    def __init__(self, layers=[2, 2, 6, 2, 2], embed_dims=[32, 64, 128, 64, 32], 
                 mlp_ratios=[2, 2, 2, 2, 2], 
                 pool_size=3, 
                 norm_layer=GroupNorm, act_layer=nn.GELU, 
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5, 
                 **kwargs):

        super().__init__()
        self.name = self.__class__.__name__
        self.simplified = False
        #skip
        self.skip1 = skip(in_channel=embed_dims[0], out_channel=embed_dims[1])
        self.relu = nn.ReLU()
        #PatchEmbedding for encoder
        self.patch_embed1 = ILM(stride=2, in_chans=3, embed_dim=embed_dims[0])
        self.patch_embed2 = ILM(stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = ILM(stride=1, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        #PatchEmbedding fo decoder
        self.patch_embed4 = UpSample(patch_size=2, stride=2, padding=0, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        self.patch_embed5 = UpSample(patch_size=2, stride=2, padding=0, in_chans=embed_dims[3], embed_dim=embed_dims[4])
        self.patch_embed6 = ILM(stride=1, in_chans=embed_dims[4], embed_dim=3)
        self.bottleneck = NLM(128)
        # set the main block in network
        self.stage1 = basic_blocks(embed_dims[0], 0, layers, 
                                 pool_size=pool_size, mlp_ratio=mlp_ratios[0],
                                 act_layer=act_layer, norm_layer=norm_layer, 
                                 drop_rate=drop_rate, 
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale, 
                                 layer_scale_init_value=layer_scale_init_value)
        self.stage2 = basic_blocks(embed_dims[1], 1, layers, 
                                 pool_size=pool_size, mlp_ratio=mlp_ratios[1],
                                 act_layer=act_layer, norm_layer=norm_layer, 
                                 drop_rate=drop_rate, 
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale, 
                                 layer_scale_init_value=layer_scale_init_value)
        self.stage3 = basic_blocks(embed_dims[2], 2, layers, 
                                 pool_size=pool_size, mlp_ratio=mlp_ratios[2],
                                 act_layer=act_layer, norm_layer=norm_layer, 
                                 drop_rate=drop_rate, 
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale, 
                                 layer_scale_init_value=layer_scale_init_value) 
        self.stage4 = basic_blocks(embed_dims[3], 3, layers, 
                                 pool_size=pool_size, mlp_ratio=mlp_ratios[3],
                                 act_layer=act_layer, norm_layer=norm_layer, 
                                 drop_rate=drop_rate, 
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale, 
                                 layer_scale_init_value=layer_scale_init_value)
        self.stage5 = basic_blocks(embed_dims[4], 4, layers, 
                                 pool_size=pool_size, mlp_ratio=mlp_ratios[4],
                                 act_layer=act_layer, norm_layer=norm_layer, 
                                 drop_rate=drop_rate, 
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale, 
                                 layer_scale_init_value=layer_scale_init_value) 
                # stores biases of surface feature branch (net simplification)
        self.register_buffer('res1_s_pre', None)
        self.register_buffer('res2_s_pre', None)
        self.register_buffer('res3_s_pre', None)

    def simplify(self, s):
        res1_s = self.patch_embed1(s)
        res1_s = self.stage1(res1_s)
        self.res1_s= res1_s
        res2_s = self.patch_embed2(res1_s)
        res2_s = self.stage2(res2_s)
        self.res2_s = res2_s
        res3_s = self.patch_embed3(res2_s)
        res3_s = self.stage3(res3_s)
        self.res3_s = res3_s

        self.res1_s_pre = self.res1_s.squeeze()
        self.res2_s_pre = self.res2_s.squeeze()
        self.res3_s_pre = self.res3_s.squeeze()

        self.simplified = True
        
    def forward(self, x, s):
        if self.simplified:
            res1_s = self.res1_s_pre
            res2_s = self.res2_s_pre
            res3_s = self.res3_s_pre
        else:
            res1_s = self.patch_embed1(s)
            res1_s = self.stage1(res1_s)
            res2_s = self.patch_embed2(res1_s)
            res2_s = self.stage2(res2_s)
            res3_s = self.patch_embed3(res2_s)
            res3_s = self.stage3(res3_s)
        
        x = self.patch_embed1(x) - res1_s 
        x0_res = self.skip1(x) 
        x = self.stage1(x)
        x = self.patch_embed2(x) - res2_s
        x = self.stage2(x)
        x = self.patch_embed3(x) - res3_s 
        x = self.stage3(x)

        x = self.bottleneck(x)
        
        x = self.patch_embed4(x) + x0_res 
        x = self.stage4(x) 
        x = self.patch_embed5(x)  
        x = self.stage5(x) 
        x = self.relu(self.patch_embed6(x)) 
        return torch.clamp(x, max=1)
 
class GRN(nn.Module):
    '''
    Implementation of GRN network.
    '''
    def __init__(self, grid_shape=(6, 6), out_size=(256, 256), with_refine=True):
        super(GRN, self).__init__()
        self.grid_shape = grid_shape
        self.out_size = out_size
        self.with_refine = with_refine 
        self.name = 'GRN' if not with_refine else 'GRN_without_refine'
        self.relu = nn.ReLU()
        self.leakyRelu = nn.LeakyReLU(0.1)

        # final refined grid
        self.register_buffer('fine_grid', None)

        # affine params
        self.affine_mat = nn.Parameter(torch.Tensor([1, 0, 0, 0, 1, 0]).view(-1, 2, 3)) # affine matrix

        # tps params
        self.nctrl = self.grid_shape[0] * self.grid_shape[1]
        self.nparam = (self.nctrl + 2) 
        ctrl_pts = pytorch_tps.uniform_grid(grid_shape) 
        self.register_buffer('ctrl_pts', ctrl_pts.view(-1, 2))
        self.theta = nn.Parameter(torch.ones((1, self.nparam * 2), dtype=torch.float32).view(-1, self.nparam, 2) * 1e-3)

        # initialization function, first checks the module type,
        def init_normal(m):
            if type(m) == nn.Conv2d:
                nn.init.normal_(m.weight, 0, 1e-4)

        # grid refinement net
        if self.with_refine:
            self.grid_refine_net = nn.Sequential(
                nn.Conv2d(2, 32, 3, 2, 1),
                self.relu,
                DeformableILM(in_chans=32, height=128, width=128),
                nn.Conv2d(32, 64, 3, 2, 1),
                self.relu,
                DeformableILM(in_chans=64, height=64, width=64),
                NonLocal(64),
                nn.ConvTranspose2d(64, 32, 2, 2, 0),
                self.relu,
                nn.ConvTranspose2d(32, 2, 2, 2, 0),
                self.leakyRelu
            )
            self.grid_refine_net.apply(init_normal)
        else:
            self.grid_refine_net = None  

    # initialize GRN's affine matrix to the input affine_vec
    def set_affine(self, affine_vec):
        self.affine_mat.data = torch.Tensor(affine_vec).view(-1, 2, 3)

    # simplify trained model to a single sampling grid for faster testing
    def simplify(self, x):
        # generate coarse affine and TPS grids
        coarse_affine_grid = F.affine_grid(self.affine_mat, torch.Size([1, x.shape[1], x.shape[2], x.shape[3]])).permute((0, 3, 1, 2)) 
        coarse_tps_grid = pytorch_tps.tps_grid(self.theta, self.ctrl_pts, (1, x.size()[1]) + self.out_size)

        # use TPS grid to sample affine grid
        tps_grid = F.grid_sample(coarse_affine_grid, coarse_tps_grid).to(device=device)

        # refine TPS grid using grid refinement net and save it to self.fine_grid
        if self.with_refine:
            self.fine_grid = torch.clamp(self.grid_refine_net(tps_grid) + tps_grid, min=-1, max=1).permute((0, 2, 3, 1))
        else:
            self.fine_grid = torch.clamp(tps_grid, min=-1, max=1).permute((0, 2, 3, 1))

    def forward(self, x):
        batch = x.shape[0]
        if self.fine_grid is None:
            # generate coarse affine and TPS grids
            coarse_affine_grid = F.affine_grid(self.affine_mat, torch.Size([1, x.shape[1], x.shape[2], x.shape[3]])).permute((0, 3, 1, 2))
            coarse_tps_grid = pytorch_tps.tps_grid(self.theta, self.ctrl_pts, (1, x.size()[1]) + self.out_size)

            # use TPS grid to sample affine grid
            tps_grid = F.grid_sample(coarse_affine_grid, coarse_tps_grid,align_corners=True).repeat(x.shape[0], 1, 1, 1)

            # refine TPS grid using grid refinement net and save it to self.fine_grid
            if self.with_refine:
                fine_grid = torch.clamp(self.grid_refine_net(tps_grid) + tps_grid, min=-1, max=1).permute((0, 2, 3, 1))
            else:
                fine_grid = torch.clamp(tps_grid, min=-1, max=1).permute((0, 2, 3, 1))
        else:
            # simplified (testing)
            fine_grid = self.fine_grid.repeat(x.shape[0], 1, 1, 1)

        # warp
        x = F.grid_sample(x, fine_grid)
        return x
    
class IDNet(nn.Module):
    def __init__(self, grn=None, pcn=None):
        super(IDNetFull, self).__init__()
        self.name = self.__class__.__name__
        # initialize from existing models or create new models
        self.grn = copy.deepcopy(grn.module) if grn is not None else GRN()
        self.pcn = copy.deepcopy(pcn.module) if pcn is not None else PCN()

    # simplify trained model to a single sampling grid for faster testing
    def simplify(self, s):
        self.grn.simplify(s)
        self.pcn.simplify(self.grn(s))

    # s is Bx3x256x256 surface image
    def forward(self, x, s):
        # geometric correction using GRN (both x and s)
        x = self.grn(x)
        s = self.grn(s)
        # photometric compensation using PCN
        x = self.pcn(x, s)

        return x

if __name__=='__main__':
    input = torch.randn(10,3,256,256)
    model = IDNet()
    output = model(input, input)
    print(output.size())
