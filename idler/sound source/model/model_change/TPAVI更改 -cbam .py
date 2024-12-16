import torch
from torch import nn
from torch.nn import functional as F

# CBAM模块定义
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        ca = self.channel_attention(x)
        x = x * ca

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(sa)
        x = x * sa

        return x

# 定义 TPAVIModule，并引入CBAM和LSTM模块
class TPAVIModule(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='dot', 
                 dimension=3, bn_layer=True):
        super(TPAVIModule, self).__init__()
        assert dimension in [1, 2, 3]
        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('mode must be one of gaussian, embedded, dot or concatenate')

        self.mode = mode
        self.dimension = dimension
        self.in_channels = in_channels
        self.inter_channels = inter_channels if inter_channels is not None else in_channels // 2
        if self.inter_channels == 0:
            self.inter_channels = 1

        self.align_channel = nn.Linear(128, in_channels)
        self.norm_layer = nn.LayerNorm(in_channels)

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        if self.mode in ['embedded', 'dot', 'concatenate']:
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        if self.mode == 'concatenate':
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )

        # CBAM 注意力模块
        self.cbam = CBAM(in_channels=self.in_channels)

        # LSTM 模块
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=in_channels, num_layers=1, batch_first=True)

    def forward(self, x, audio=None):
        audio_temp = 0
        batch_size, C = x.size(0), x.size(1)
        if audio is not None:
            H, W = x.shape[-2], x.shape[-1]
            audio_temp = self.align_channel(audio)
            audio = audio_temp.permute(0, 2, 1)
            audio = audio.unsqueeze(-1).unsqueeze(-1)
            audio = audio.repeat(1, 1, 1, H, W)
        else:
            audio = x

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = audio.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)
        elif self.mode in ["embedded", "dot"]:
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(audio).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)
        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(audio).view(batch_size, self.inter_channels, 1, -1)
            h, w = theta_x.size(2), phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)
            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat).view(batch_size, concat.size(2), concat.size(3))

        if self.mode in ["gaussian", "embedded"]:
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode in ["dot", "concatenate"]:
            f_div_C = f / f.size(-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W_z(y)
        z = W_y + x

        # 应用CBAM模块
        z = self.cbam(z)

        # LSTM时序处理
        z = z.permute(0, 2, 3, 4, 1)  # [bs, T, H, W, C]
        z, _ = self.lstm(z.view(batch_size, -1, C))  # LSTM操作
        z = z.view(batch_size, C, *x.size()[2:])

        z = self.norm_layer(z.permute(0, 4, 1, 2, 3))

        return z, audio_temp
