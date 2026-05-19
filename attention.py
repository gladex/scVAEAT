import torch
import torch.nn as nn
import torch.nn.functional as F


class ATTN(nn.Module):
    def __init__(self, dff_size, model_size, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(model_size, dff_size)
        self.act = F.gelu
        self.fc2 = nn.Linear(dff_size, dff_size)
        self.attn = AttnModule2(dff_size)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.fc3 = nn.Linear(dff_size, model_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = x.unsqueeze(1)
        x = self.attn(x)
        x = x.squeeze(1)
        x = self.drop(x)
        x = self.fc3(x)
        return x


class SPA(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool1d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool1d(2)
        self.avg_pool4 = nn.AdaptiveAvgPool1d(4)
        self.fc = nn.Sequential(
            nn.Linear(channel * (1 + 2 + 4), channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, l, c = x.shape
        x = x.transpose(1, 2)
        y1 = self.avg_pool1(x).reshape(b, -1)
        y2 = self.avg_pool2(x).reshape(b, -1)
        y3 = self.avg_pool4(x).reshape(b, -1)
        y = torch.cat([y1, y2, y3], dim=1)
        y = self.fc(y).unsqueeze(1)
        return y


class AttnModule2(nn.Module):
    def __init__(self, model_size, act_ratio=0.25, act_fn=F.gelu, gate_fn=torch.sigmoid):
        super().__init__()
        reduce_channels = int(model_size * act_ratio)
        self.norm = nn.LayerNorm(model_size)
        self.global_reduce = nn.Linear(model_size, reduce_channels)
        self.local_reduce = nn.Linear(model_size, reduce_channels)
        self.act_fn = act_fn
        self.spatial_select = nn.Linear(2 * reduce_channels, 1)
        self.gate_fn = gate_fn
        self.spa = SPA(model_size)

    def forward(self, x):
        ori_x = x
        x = self.norm(x)
        x_global = x.mean(dim=1, keepdim=True)
        x_global = self.act_fn(self.global_reduce(x_global))
        x_local = self.act_fn(self.local_reduce(x))
        c_attn = self.spa(x)
        s_attn_input = torch.cat([x_local, x_global.repeat(1, x.size(1), 1)], dim=-1)
        s_attn = self.spatial_select(s_attn_input)
        s_attn = self.gate_fn(s_attn)
        attn = c_attn * s_attn
        return ori_x * attn
