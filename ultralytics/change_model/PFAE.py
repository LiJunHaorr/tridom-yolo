# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import Softmax
# from einops import rearrange, repeat
#
#
# def custom_complex_normalization(input_tensor, dim=-1):
#     """自定义复数归一化函数
#     对复数的实部和虚部分别进行softmax归一化
#
#     参数:
#         input_tensor: 复数张量
#         dim: 归一化维度
#
#     返回:
#         归一化后的复数张量
#     """
#     real_part = input_tensor.real
#     imag_part = input_tensor.imag
#     norm_real = F.softmax(real_part, dim=dim)
#     norm_imag = F.softmax(imag_part, dim=dim)
#
#     normalized_tensor = torch.complex(norm_real, norm_imag)
#
#     return normalized_tensor
#
#
# class PFAE(nn.Module):
#     """PFAE (Parallel Frequency Attention Enhancement) 模块
#     用于伪装目标检测的频率增强注意力网络
#
#     参数:
#         dim: 输入通道数
#         in_dim: 内部特征维度
#     """
#
#     def __init__(self, dim, in_dim):
#         super(PFAE, self).__init__()
#         # 下采样卷积
#         self.down_conv = nn.Sequential(
#             nn.Conv2d(dim, in_dim, 3, padding=1),
#             nn.BatchNorm2d(in_dim),
#             nn.ReLU(True)
#         )
#         down_dim = in_dim // 2  # 下采样后的特征维度
#
#         # 五个并行卷积分支，具有不同的膨胀率
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_dim, down_dim, kernel_size=1),
#             nn.BatchNorm2d(down_dim),
#             nn.ReLU(True)
#         )
#
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=3, padding=3),
#             nn.BatchNorm2d(down_dim),
#             nn.ReLU(True)
#         )
#         # 频率注意力相关参数
#         self.query_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
#         self.key_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
#         self.value_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
#         self.gamma2 = nn.Parameter(torch.zeros(1))  # 可学习的注意力权重
#
#         # 其他卷积分支（类似结构）
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=5, padding=5),
#             nn.BatchNorm2d(down_dim),
#             nn.ReLU(True)
#         )
#         self.query_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
#         self.key_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
#         self.value_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
#         self.gamma3 = nn.Parameter(torch.zeros(1))
#
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=7, padding=7),
#             nn.BatchNorm2d(down_dim),
#             nn.ReLU(True)
#         )
#         self.query_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
#         self.key_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
#         self.value_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
#         self.gamma4 = nn.Parameter(torch.zeros(1))
#
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=9, padding=9),
#             nn.BatchNorm2d(down_dim),
#             nn.ReLU(True)
#         )
#         self.query_conv5 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
#         self.key_conv5 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
#         self.value_conv5 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
#         self.gamma5 = nn.Parameter(torch.zeros(1))
#
#         # 全局平均池化分支
#         self.conv6 = nn.Sequential(
#             nn.Conv2d(in_dim, down_dim, kernel_size=1),
#             nn.BatchNorm2d(down_dim),
#             nn.ReLU(True)
#         )
#
#         # 特征融合
#         self.fuse = nn.Sequential(
#             nn.Conv2d(6 * down_dim, down_dim, kernel_size=1),
#             nn.BatchNorm2d(down_dim),
#             nn.ReLU(True)
#         )
#
#         # 输出层
#         self.out = nn.Sequential(
#             nn.Conv2d(down_dim, down_dim // 2, kernel_size=3, padding=1),
#             nn.BatchNorm2d(down_dim // 2),
#             nn.ReLU(True),
#             nn.Conv2d(down_dim // 2, 1, kernel_size=1)
#         )
#
#         # 注意力温度参数
#         self.temperature = nn.Parameter(torch.ones(8, 1, 1))
#         self.project_out = nn.Conv2d(down_dim * 2, down_dim, kernel_size=1, bias=False)
#
#         # 注意力权重生成
#         self.weight = nn.Sequential(
#             nn.Conv2d(down_dim, down_dim // 16, 1, bias=True),
#             nn.BatchNorm2d(down_dim // 16),
#             nn.ReLU(True),
#             nn.Conv2d(down_dim // 16, down_dim, 1, bias=True),
#             nn.Sigmoid())
#
#         self.softmax = Softmax(dim=-1)
#         self.norm = nn.BatchNorm2d(down_dim)
#         self.relu = nn.ReLU(True)
#         self.num_heads = 8  # 多头注意力头数
#
#     def forward(self, x):
#         """前向传播
#
#         参数:
#             x: 输入特征图 [B, C, H, W]
#
#         返回:
#             输出特征图 [B, 1, H, W]
#         """
#         x = self.down_conv(x)  # 初始下采样
#         conv1 = self.conv1(x)  # 1x1卷积分支
#
#         # 第一个膨胀卷积分支
#         conv2 = self.conv2(x)
#         b, c, h, w = conv2.shape
#
#         # 频率域注意力计算
#         q_f_2 = torch.fft.fft2(conv2.float())
#         k_f_2 = torch.fft.fft2(conv2.float())
#         v_f_2 = torch.fft.fft2(conv2.float())
#         tepqkv = torch.fft.fft2(conv2.float())
#
#         # 多头注意力处理
#         q_f_2 = rearrange(q_f_2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k_f_2 = rearrange(k_f_2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v_f_2 = rearrange(v_f_2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#
#         # 归一化处理
#         q_f_2 = torch.nn.functional.normalize(q_f_2, dim=-1)
#         k_f_2 = torch.nn.functional.normalize(k_f_2, dim=-1)
#
#         # 注意力计算
#         attn_f_2 = (q_f_2 @ k_f_2.transpose(-2, -1)) * self.temperature
#         attn_f_2 = custom_complex_normalization(attn_f_2, dim=-1)
#
#         # 反变换回空间域
#         out_f_2 = torch.abs(torch.fft.ifft2(attn_f_2 @ v_f_2))
#         out_f_2 = rearrange(out_f_2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
#
#         # 局部注意力增强
#         out_f_l_2 = torch.abs(torch.fft.ifft2(self.weight(tepqkv.real) * tepqkv))
#
#         # 特征融合
#         out_2 = self.project_out(torch.cat((out_f_2, out_f_l_2), 1))
#         F_2 = torch.add(out_2, conv2)  # 残差连接
#
#         # 其他分支处理（类似结构）
#         conv3 = self.conv3(x + F_2)
#         b, c, h, w = conv3.shape
#
#         q_f_3 = torch.fft.fft2(conv3.float())
#         k_f_3 = torch.fft.fft2(conv3.float())
#         v_f_3 = torch.fft.fft2(conv3.float())
#         tepqkv = torch.fft.fft2(conv3.float())
#
#         q_f_3 = rearrange(q_f_3, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k_f_3 = rearrange(k_f_3, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v_f_3 = rearrange(v_f_3, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#
#         q_f_3 = torch.nn.functional.normalize(q_f_3, dim=-1)
#         k_f_3 = torch.nn.functional.normalize(k_f_3, dim=-1)
#         attn_f_3 = (q_f_3 @ k_f_3.transpose(-2, -1)) * self.temperature
#         attn_f_3 = custom_complex_normalization(attn_f_3, dim=-1)
#         out_f_3 = torch.abs(torch.fft.ifft2(attn_f_3 @ v_f_3))
#         out_f_3 = rearrange(out_f_3, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
#         out_f_l_3 = torch.abs(torch.fft.ifft2(self.weight(tepqkv.real) * tepqkv))
#         out_3 = self.project_out(torch.cat((out_f_3, out_f_l_3), 1))
#         F_3 = torch.add(out_3, conv3)
#
#         # 剩余分支处理...
#         conv4 = self.conv4(x + F_3)
#         b, c, h, w = conv4.shape
#
#         q_f_4 = torch.fft.fft2(conv4.float())
#         k_f_4 = torch.fft.fft2(conv4.float())
#         v_f_4 = torch.fft.fft2(conv4.float())
#         tepqkv = torch.fft.fft2(conv4.float())
#
#         q_f_4 = rearrange(q_f_4, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k_f_4 = rearrange(k_f_4, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v_f_4 = rearrange(v_f_4, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#
#         q_f_4 = torch.nn.functional.normalize(q_f_4, dim=-1)
#         k_f_4 = torch.nn.functional.normalize(k_f_4, dim=-1)
#         attn_f_4 = (q_f_4 @ k_f_4.transpose(-2, -1)) * self.temperature
#         attn_f_4 = custom_complex_normalization(attn_f_4, dim=-1)
#         out_f_4 = torch.abs(torch.fft.ifft2(attn_f_4 @ v_f_4))
#         out_f_4 = rearrange(out_f_4, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
#         out_f_l_4 = torch.abs(torch.fft.ifft2(self.weight(tepqkv.real) * tepqkv))
#         out_4 = self.project_out(torch.cat((out_f_4, out_f_l_4), 1))
#         F_4 = torch.add(out_4, conv4)
#
#         conv5 = self.conv5(x + F_4)
#         b, c, h, w = conv5.shape
#
#         q_f_5 = torch.fft.fft2(conv5.float())
#         k_f_5 = torch.fft.fft2(conv5.float())
#         v_f_5 = torch.fft.fft2(conv5.float())
#         tepqkv = torch.fft.fft2(conv5.float())
#
#         q_f_5 = rearrange(q_f_5, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k_f_5 = rearrange(k_f_5, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v_f_5 = rearrange(v_f_5, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#
#         q_f_5 = torch.nn.functional.normalize(q_f_5, dim=-1)
#         k_f_5 = torch.nn.functional.normalize(k_f_5, dim=-1)
#         attn_f_5 = (q_f_5 @ k_f_5.transpose(-2, -1)) * self.temperature
#         attn_f_5 = custom_complex_normalization(attn_f_5, dim=-1)
#         out_f_5 = torch.abs(torch.fft.ifft2(attn_f_5 @ v_f_5))
#         out_f_5 = rearrange(out_f_5, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
#         out_f_l_5 = torch.abs(torch.fft.ifft2(self.weight(tepqkv.real) * tepqkv))
#         out_5 = self.project_out(torch.cat((out_f_5, out_f_l_5), 1))
#         F_5 = torch.add(out_5, conv5)
#
#         # 全局平均池化分支
#         conv5 = F.upsample(
#             self.conv6(F.adaptive_avg_pool2d(x, 1)),
#             size=x.size()[2:],
#             mode='bilinear'
#         )
#
#         # 所有分支特征融合
#         F_out = self.out(self.fuse(torch.cat((conv1, F_2, F_3, F_4, F_5, conv5), 1)))
#
#         return F_out
#
#
# def main():
#     """测试PFAE模块的主函数"""
#     # 设置测试参数
#     batch_size = 4
#     channels = 64
#     height = 32
#     width = 32
#
#     # 创建随机输入张量
#     x = torch.randn(batch_size, channels, height, width)
#     print(f"输入形状: {x.shape}")
#
#     # 初始化PFAE模块
#     pfae = PFAE(dim=channels, in_dim=32)
#     print("\nPFAE模块结构:")
#     print(pfae)
#
#     # 前向传播
#     output = pfae(x)
#     print(f"\n输出形状: {output.shape} (应为[batch, 1, height, width])")
#
#     # 验证输出形状
#     assert output.shape == (batch_size, 1, height, width), "输出形状不正确"
#     print("\n测试通过! 输出形状符合预期")
#
#
# if __name__ == "__main__":
#     main()


import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

# https://arxiv.org/pdf/2503.11030


def custom_complex_normalization(input_tensor, dim=-1):
    """自定义复数归一化函数."""
    real_part = input_tensor.real
    imag_part = input_tensor.imag
    norm_real = F.softmax(real_part, dim=dim)
    norm_imag = F.softmax(imag_part, dim=dim)
    return torch.complex(norm_real, norm_imag)


class PFAE(nn.Module):
    """PFAE (Parallel Frequency Attention Enhancement) 模块."""

    def __init__(self, dim):
        super().__init__()

        in_dim = dim // 2
        # 下采样层
        self.down_conv = nn.Sequential(nn.Conv2d(dim, in_dim, 3, padding=1), nn.BatchNorm2d(in_dim), nn.ReLU(True))
        down_dim = in_dim // 2

        # 五个并行卷积分支
        self.conv1 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.ReLU(True))

        # 膨胀卷积分支（dilation=3）
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=3, padding=3), nn.BatchNorm2d(down_dim), nn.ReLU(True)
        )

        # 膨胀卷积分支（dilation=5）
        self.conv3 = nn.Sequential(
            nn.Conv2d(down_dim, down_dim, kernel_size=3, dilation=5, padding=5),  # 修改输入通道为down_dim
            nn.BatchNorm2d(down_dim),
            nn.ReLU(True),
        )

        # 膨胀卷积分支（dilation=7）
        self.conv4 = nn.Sequential(
            nn.Conv2d(down_dim, down_dim, kernel_size=3, dilation=7, padding=7),  # 修改输入通道为down_dim
            nn.BatchNorm2d(down_dim),
            nn.ReLU(True),
        )

        # 膨胀卷积分支（dilation=9）
        self.conv5 = nn.Sequential(
            nn.Conv2d(down_dim, down_dim, kernel_size=3, dilation=9, padding=9),  # 修改输入通道为down_dim
            nn.BatchNorm2d(down_dim),
            nn.ReLU(True),
        )

        # 全局平均池化分支
        self.conv6 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.ReLU(True))

        # 特征融合层
        self.fuse = nn.Sequential(
            nn.Conv2d(6 * down_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.ReLU(True)
        )

        # 输出层
        self.out = nn.Sequential(
            nn.Conv2d(down_dim, down_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(down_dim // 2),
            nn.ReLU(True),
            nn.Conv2d(down_dim // 2, dim, kernel_size=1),
        )

        # 注意力参数
        self.temperature = nn.Parameter(torch.ones(8, 1, 1))
        self.project_out = nn.Conv2d(down_dim * 2, down_dim, kernel_size=1, bias=False)

        # 注意力权重生成
        self.weight = nn.Sequential(
            nn.Conv2d(down_dim, down_dim // 8, 1, bias=True),
            nn.BatchNorm2d(down_dim // 8),
            nn.ReLU(True),
            nn.Conv2d(down_dim // 8, down_dim, 1, bias=True),
            nn.Sigmoid(),
        )

        self.num_heads = 8

    def frequency_attention(self, x):
        """频率注意力计算."""
        _b, _c, h, w = x.shape
        # 保存原始数据类型
        original_dtype = x.dtype

        q_f = torch.fft.fft2(x.float())
        k_f = torch.fft.fft2(x.float())
        v_f = torch.fft.fft2(x.float())
        tepqkv = torch.fft.fft2(x.float())

        # 多头处理
        q_f = rearrange(q_f, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k_f = rearrange(k_f, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v_f = rearrange(v_f, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        # 归一化
        q_f = F.normalize(q_f, dim=-1)
        k_f = F.normalize(k_f, dim=-1)

        # 注意力计算
        attn_f = (q_f @ k_f.transpose(-2, -1)) * self.temperature
        attn_f = custom_complex_normalization(attn_f, dim=-1)

        # 反变换
        out_f = torch.abs(torch.fft.ifft2(attn_f @ v_f))
        out_f = rearrange(out_f, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)

        # 局部注意力 - 确保类型匹配
        tepqkv_real = tepqkv.real.to(original_dtype)  # 转换为原始数据类型
        weight_out = self.weight(tepqkv_real)
        out_f_l = torch.abs(torch.fft.ifft2(weight_out.to(tepqkv.dtype) * tepqkv))

        # 特征融合 - 确保类型一致
        out_f = out_f.to(original_dtype)
        out_f_l = out_f_l.to(original_dtype)
        out = self.project_out(torch.cat((out_f, out_f_l), 1))
        return out + x  # 残差连接

    def forward(self, x):
        """前向传播."""
        x = self.down_conv(x)  # [B, in_dim, H, W]
        conv1 = self.conv1(x)  # [B, down_dim, H, W]

        # 各分支处理
        F_2 = self.frequency_attention(self.conv2(x))
        F_3 = self.frequency_attention(self.conv3(F_2))  # 直接使用F_2作为输入
        F_4 = self.frequency_attention(self.conv4(F_3))
        F_5 = self.frequency_attention(self.conv5(F_4))

        # 全局分支
        conv5 = F.interpolate(self.conv6(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode="bilinear")

        # 特征融合
        F_out = self.out(self.fuse(torch.cat((conv1, F_2, F_3, F_4, F_5, conv5), 1)))
        return F_out


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = x.to(self.conv.weight.device)  # 确保输入张量在卷积层所在设备
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = x.to(self.conv.weight.device)  # 确保输入张量在卷积层所在设备
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class Bottleneck_PFAE(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = PFAE(c_)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_PFAE(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


# 在c3k=True时，使用Bottleneck_HSMSSD特征融合，为false的时候我们使用普通的Bottleneck提取特征
class C3k2_PFAE(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


def main():
    """测试函数."""
    # 测试参数
    batch_size = 2
    channels = 64
    # 创建输入
    x = torch.randn(batch_size, channels, 8, 8)
    print(f"输入形状: {x.shape}")

    # 初始化模块
    pfae = PFAE(dim=channels)
    print("\n模块参数量:", sum(p.numel() for p in pfae.parameters()))

    # 前向传播
    output = pfae(x)
    print(f"\n输出形状: {output.shape}")


if __name__ == "__main__":
    main()
