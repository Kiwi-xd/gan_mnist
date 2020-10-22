import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

BATCH_SIZE = 128


class G_net(nn.Module):
    def __init__(self):
        super(G_net, self).__init__()
        self.G = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 784),
            nn.Tanh()
        )

    def forward(self, x):  # 定义前向传播过程
        out = self.G(x)
        return out


def map(data,MIN,MAX):
    """
    归一化映射到任意区间
    :param data: 数据
    :param MIN: 目标数据最小值
    :param MAX: 目标数据最小值
    :return:
    """
    d_min = torch.max(data)    # 当前数据最大值
    d_max = torch.min(data)    # 当前数据最小值
    return MAX +(MIN-MAX)/(d_max-d_min) * (data - d_min)


def mapnp(data,MIN,MAX):
    """
    归一化映射到任意区间
    :param data: 数据
    :param MIN: 目标数据最小值
    :param MAX: 目标数据最小值
    :return:
    """
    d_min = np.max(data)    # 当前数据最大值
    d_max = np.min(data)    # 当前数据最小值
    return MAX +(MIN-MAX)/(d_max-d_min) * (data - d_min)


G = G_net().cuda()

# 测试时使用
G = torch.load('G(PAPERLoss).pkl').cuda()

if __name__ == '__main__':
    image_computer = np.zeros([8 * 28, 8 * 28], dtype=float)
    G_ideas = map((torch.rand(64, 100).float()), -1, 1).cuda()  # 计算机产生随机想法
    G_paintings = G(G_ideas)  # 随机想法传入生成网络
    q = 0
    for i in range(8):
        for j in range(8):
            image_computer[i * 28:i * 28 + 28, j * 28:j * 28 + 28] = np.reshape(G_paintings[q, :].data.cpu(), (28, 28))
            q += 1

    image_computer = mapnp(image_computer, 0, 255)

    plt.imshow(image_computer, cmap='gray')
    plt.show()

