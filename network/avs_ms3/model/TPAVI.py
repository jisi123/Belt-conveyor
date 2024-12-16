import torch
from torch import nn
from torch.nn import functional as F

#定义一个新的 PyTorch 模块类 TPAVIModule，它继承自 nn.Module。
class TPAVIModule(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='dot', 
                 dimension=3, bn_layer=True):
        """
        args:
            in_channels: original channel size (1024 in the paper)原始通道
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            如果未指定，块内的通道尺寸将减少到一半
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            支持高斯、嵌入式高斯、点积和串联
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            可以是 1（时间）、2（空间）、3（时空）
            bn_layer: whether to add batch norm
        """
        #调用父类 nn.Module 的构造函数
        super(TPAVIModule, self).__init__()
        #断言 dimension 参数必须是 1、2 或 3。
        assert dimension in [1, 2, 3]
        #检查 mode 参数是否是支持的模式之一。
        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')
        #将传入的模式和维度设置为实例变量。设置输入和内部通道数。
        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # 通道大小在块内减小到一半
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        
        # 添加对齐通道
        #定义一个线性层，用于对音频数据进行转换。
        #定义一个层归一化层
        self.align_channel = nn.Linear(128, in_channels)
        self.norm_layer=nn.LayerNorm(in_channels)

        # 为不同的维度分配适当的卷积、最大池和批处理范数层,根据维度设置相应的卷积层、最大池化层和批量归一化层。
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

        # 论文中的函数 g，通过核大小为 1 的转换
        #定义一个卷积层 g，用于通过 1x1 卷积操作。
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
#如果需要批量归一化层，则定义 W_z 并初始化其权重和偏置
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

        # define theta and phi for all operations except gaussian
        #如果模式是嵌入式、点积或连接，则定义 theta 和 phi 卷积层
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        #如果模式是连接，则定义 W_f 序列，用于连接操作。
        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                    nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                    nn.ReLU()
                )

    #定义模块的前向传播方法。
    def forward(self, x, audio=None):
        """
        args:
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
            audio: (N, T, C)维度
        """
#audio_temp = 0：初始化一个临时变量来存储音频数据。获取输入数据的批量大小和通道数。如果提供了音频数据，则进行处理。


        audio_temp = 0
        batch_size, C = x.size(0), x.size(1)
        if audio is not None:
            # print('==> audio.shape', audio.shape)
            H, W = x.shape[-2], x.shape[-1]
            audio_temp = self.align_channel(audio) # [bs, T, C]
            audio = audio_temp.permute(0, 2, 1) # [bs, C, T]
            audio = audio.unsqueeze(-1).unsqueeze(-1) # [bs, C, T, 1, 1]
            audio = audio.repeat(1, 1, 1, H, W) # [bs, C, T, H, W]
        else:
            audio = x

        # (N, C, THW)通过 g 卷积层处理输入数据 x。
        g_x = self.g(x).view(batch_size, self.inter_channels, -1) # [bs, C, THW]
        # print('g_x.shape', g_x.shape)
        # g_x = x.view(batch_size, C, -1)  # [bs, C, THW]
        g_x = g_x.permute(0, 2, 1) # [bs, THW, C]重新排列数据维度
#根据不同的模式计算特征图 f
        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = audio.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1) # [bs, C', THW]
            phi_x = self.phi(audio).view(batch_size, self.inter_channels, -1) # [bs, C', THW]
            theta_x = theta_x.permute(0, 2, 1) # [bs, THW, C']
            f = torch.matmul(theta_x, phi_x) # [bs, THW, THW]

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(audio).view(batch_size, self.inter_channels, 1, -1)
            
            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)
            
            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))
        #如果模式是高斯或嵌入式，则对 f 应用 softmax
        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1) # number of position in x
            f_div_C = f / N  # [bs, THW, THW]
        
        y = torch.matmul(f_div_C, g_x) # [bs, THW, C]
        
        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous() # [bs, C, THW]重新排列并确保数据连续。
        y = y.view(batch_size, self.inter_channels, *x.size()[2:]) # [bs, C', T, H, W]将 y 重塑为正确的形状
        
        W_y = self.W_z(y)  # [bs, C, T, H, W]通过 W_z 卷积层处理 y。
        # residual connection
        z = W_y + x #  # [bs, C, T, H, W]计算残差连接

        # add LayerNorm重新排列数据维度
        z =  z.permute(0, 2, 3, 4, 1) # [bs, T, H, W, C]
        z = self.norm_layer(z)
        z = z.permute(0, 4, 1, 2, 3) # [bs, C, T, H, W]
        
        return z, audio_temp#：返回处理后的数据和音频临时变量。


