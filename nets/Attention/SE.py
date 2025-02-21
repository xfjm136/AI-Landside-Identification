import torch
from torch import nn
from torch.nn import init


class SEAttention(nn.Module):
    def __init__(self, channel=None, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 第一个全连接层
            nn.ReLU(inplace=True),  # 激活函数
            nn.Linear(channel // reduction, channel, bias=False),  # 第二个全连接层
            nn.Sigmoid()  # Sigmoid 激活函数
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # print(f"Input shape: {x.shape}")
        y = self.avg_pool(x).view(b, c)
        # print(f"Shape after avg_pool: {y.shape}")
        y = self.fc(y).view(b, c, 1, 1)
        # print(f"Shape after fc: {y.shape}")
        return x * y.expand_as(x)


if __name__ == '__main__':
    input = torch.randn(2, 40, 256, 256)  # 输入张量 (b=2, c=40, h=256, w=256)
    se = SEAttention(channel=40, reduction=16)  # 创建 SEAttention 实例
    output = se(input)  # 前向传播
    print(output.shape)  # 打印输出张量形状
