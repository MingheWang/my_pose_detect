import torch
import torch.nn as nn
import numpy as np
from .pose_layers import Conv, Hourglass, Pool, Residual
"""
#############################################################################
model_input_size:
        person_feature_map : (num_person , chennal , fmap_size , fmap_size)
#############################################################################
"""

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 256, 4, 4)

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)
    
class PoseNet(nn.Module):
    def __init__(self, nstack=2, in_dim=768, out_dim=17, bn=False, increase=0, **kwargs):
        # nstack代表hourglass模块的数目
        # increase为递归生产每个hourglass模块的时候，通道的增加数目
        super(PoseNet, self).__init__()
        
        self.nstack = nstack
        # 上采样，(K,768,14,14) ---> (K,384,28,28) ---> (K,192,56,56)
        n_ch_hg = in_dim//4
        self.pre_process = nn.Sequential(
            nn.Upsample(scale_factor=2,mode="nearest"),
            Residual(in_dim, in_dim//2),
            nn.Upsample(scale_factor=2,mode="nearest"),
            Residual(in_dim//2, n_ch_hg)
        )
        
        self.hgs = nn.ModuleList([
        nn.Sequential(
            Hourglass(4, n_ch_hg, bn, increase),
        ) for i in range(nstack)])                  # 4：代表每个沙漏模块的递归数目
        
        self.features = nn.ModuleList( [
        nn.Sequential(
            Residual(n_ch_hg, n_ch_hg),
            Conv(n_ch_hg, n_ch_hg, 1, bn=True, relu=True)
        ) for i in range(nstack)] )
        
        self.outs = nn.ModuleList( [Conv(n_ch_hg, out_dim, 1, relu=False, bn=False) for i in range(nstack)] )
        self.merge_features = nn.ModuleList( [Merge(n_ch_hg, n_ch_hg) for i in range(nstack-1)] )
        self.merge_preds = nn.ModuleList( [Merge(out_dim, n_ch_hg) for i in range(nstack-1)] )
        self.nstack = nstack

    def forward(self, input_features):
        # 上采样，(K,768,14,14) ---> (K,384,28,28) ---> (K,192,56,56)
        x = self.pre_process(input_features)
        for i in range(self.nstack):
            hg = self.hgs[i](x)                 # 192 ---> 192
            feature = self.features[i](hg)      # 192 ---> 192
            preds = self.outs[i](feature)       # 192 ---> out_dim  # (K,out_dim,56,56)     
            # (K,192,56,56) + (K,192,56,56) + (K,192,56,56)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        return preds


######################################################################################################################

