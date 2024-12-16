import torch
from torch import nn
from torch.nn import functional as F

class TPAVIModule(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='dot', 
                 dimension=3, bn_layer=True):
        """
        args:
            in_channels: original channel size (1024 in the paper)原始通道
            inter_channels: channel size inside the block if not specified, reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(TPAVIModule, self).__init__()
        assert dimension in [1, 2, 3]
        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')
        
        self.mode = mode
        self.dimension = dimension
        self.in_channels = in_channels
        self.inter_channels = inter_channels or in_channels // 2
        if self.inter_channels == 0:
            self.inter_channels = 1
        
        # 添加对齐通道，增加ReLU激活
        self.align_channel = nn.Sequential(
            nn.Linear(128, in_channels),
            nn.ReLU()
        )
        self.norm_layer = nn.LayerNorm(in_channels)

        # 根据维度选择卷积和池化层
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

        # 定义卷积层g
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # 批量归一化层
        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

        # 定义theta和phi
        if self.mode in ["embedded", "dot", "concatenate"]:
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, x, audio=None):
        """
        args:
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
            audio: (N, T, C)维度
        """
        batch_size, C = x.size(0), x.size(1)
        
        # 处理音频
        if audio is not None:
            H, W = x.shape[-2], x.shape[-1]
            audio_temp = self.align_channel(audio)
            audio = audio_temp.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
            audio = audio.repeat(1, 1, 1, H, W)
        else:
            audio = x

        # g_x计算
        g_x = self.g(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1).permute(0, 2, 1)
            phi_x = audio.view(batch_size, self.in_channels, -1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode in ["embedded", "dot"]:
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
            phi_x = self.phi(audio).view(batch_size, self.inter_channels, -1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(audio).view(batch_size, self.inter_channels, 1, -1)
            h, w = theta_x.size(2), phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)
            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat).view(concat.size(0), concat.size(2), concat.size(3))

        # 计算f_div_C
        if self.mode in ["gaussian", "embedded"]:
            f_div_C = F.softmax(f, dim=-1)
        else:
            N = f.size(-1)
            f_div_C = f / N

        y = torch.matmul(f_div_C, g_x).permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        
        W_y = self.W_z(y)
        
        # 残差连接，添加可学习比例alpha
        alpha = 0.5
        z = alpha * W_y + (1 - alpha) * x

        # 添加LayerNorm
        z = z.permute(0, 2, 3, 4, 1)
        z = self.norm_layer(z)
        z = z.permute(0, 4, 1, 2, 3)

        return z, audio_temp if audio is not None else None
