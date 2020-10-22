import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from mnist_processing import load_train_images  # 解析并加载mnist数据集
import os

BATCH_SIZE = 128
EPOCH = 2000  # 设置训练epoch

result_path = './train_result(PAPERLoss)' # 存储训练结果的文件夹路径


class G_net(nn.Module):  # 定义生成器
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


class D_net(nn.Module):  # 定义判别器
    def __init__(self):
        super(D_net, self).__init__()
        self.D = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):  # 定义前向传播过程
        out = self.D(x)
        return out


def maptorch(data, MIN, MAX):  # 定义maptorch用于归一化映射到任意区间，输入data为tensor变量
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


def mapnp(data, MIN ,MAX):   # 定义mapnp用于归一化映射到任意区间，输入data为numpy变量
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


# 将生成器和判别器分别传给G和D，并传入GPU
G = G_net().cuda()
D = D_net().cuda()

# 如需接着训练已保存网络则加载相应的网络
# G = torch.load('G3.pkl').cuda()
# D = torch.load('D3.pkl').cuda()

if __name__ == '__main__':
    train_images = load_train_images()  # 加载数据集图片，存入train_images中

    # 定义优化器
    opt_D = torch.optim.Adam(D.parameters(), lr=0.0001)
    opt_G = torch.optim.Adam(G.parameters(), lr=0.0001)


    # 定义真假标签
    real_label = torch.ones(BATCH_SIZE).cuda()
    fake_label = torch.zeros(BATCH_SIZE).cuda()

    ITERATIONS = 468  # 每个epoch中迭代mini-batch的次数

    m = BATCH_SIZE*ITERATIONS  # 共BATCH_SIZE*ITERATIONS个样本

    print(f'共使用{m}个样本进行训练')

    # 新建real_image用于存放一个mini-batch的真实样本
    real_image = torch.tensor(np.zeros((BATCH_SIZE, 28*28), dtype=float)).float().cuda()

    # real_show用于展示真实图片样本，展示为尺寸为（8*28,8*28）的一张图片。
    real_show = np.zeros([8*28, 8*28], dtype=float)
    k = 0
    for i in range(8):
        for j in range(8):
            real_show[i*28:i*28+28, j*28:j*28+28] = train_images[k]
            k += 1

    # 将real_show存储为一张图片，存储到指定的文件夹中，名为real_img
    plt.imshow(real_show, cmap='gray')
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    plt.savefig(f'{result_path}/real_img.jpg')

    for epoch in range(EPOCH):       # 训练1000epoch
        for iterations in range(ITERATIONS):
            for i in range(BATCH_SIZE):
                real_image[i, :] = torch.tensor(np.reshape(train_images[iterations*BATCH_SIZE+i], (28*28))).float().cuda()  # real_image存储一个Batch的真实样本

            real_image = maptorch(real_image, -1, 1)  # 将真实样本归一化到-1,1

            # # 将真假标签都传入GPU
            # real_label = torch.ones(BATCH_SIZE, 1).cuda()
            # fake_label = torch.zeros(BATCH_SIZE, 1).cuda()

            # 计算机产生随机想法，生成尺寸为(BATCH_SIZE, 100)的，范围为（-1,1）符合均匀分布的随机数
            G_ideas = (maptorch(torch.rand(BATCH_SIZE, 100).float(), -1, 1)).cuda()

            fake_image = G(G_ideas)  # 随机想法传入生成假样本

            # # 计算判别器对于假样本和真样本的loss值，使用nn模块中的二分类交叉熵进行计算
            # d_loss_fake = nn.BCELoss()(D(fake_image), fake_label)
            # d_loss_real = nn.BCELoss()(D(real_image), real_label)

            # # 计算判别器的loss值
            # D_loss = d_loss_real + d_loss_fake

            prob_artist0 = D(real_image)  # 对专业人士作的曲线的评价
            prob_artist1 = D(fake_image)  # 对计算机作的曲线的评价

            # 判别器的损失函数，负号代表优化过程使整个式子朝着变大的方向优化，也就是增加判别器对专业人士的评价
            D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(torch.tensor(1).float() - prob_artist1))

            # 优化判别器
            opt_D.zero_grad()
            D_loss.backward(retain_graph=True)  # retain_graph 这个参数是为了再次使用计算的值去训练生成器
            opt_D.step()

            G_ideas2 = (torch.rand(BATCH_SIZE, 100).float()).cuda()  # 计算机再次产生随机想法
            fake_image2 = G(G_ideas)  # 随机想法传入生成另一组假样本
            prob_artist3 = D(fake_image2)  # 对计算机作的曲线的评价
            # 生成器的损失函数，使整个式子朝着变小的方向优化，也就是想要提高判别器对计算机的评价
            G_loss = torch.mean(torch.log(torch.tensor(1).float() - prob_artist3))


            # 优化生成器
            opt_G.zero_grad()
            G_loss.backward()
            opt_G.step()

        G_ideas3 = maptorch((torch.rand(BATCH_SIZE, 100).float()), -1, 1).cuda()  # 计算机再次产生随机想法
        fake_image3 = G(G_ideas)  # 随机想法传入生成假样本fake_image3

        # 打印训练过程中的相关参数
        print(f'epoch:{epoch + 1}/{EPOCH},D_loss:{D_loss.cpu().detach().numpy()}, '
              f'G_loss:{G_loss.cpu().detach().numpy()},'
              f'D_real:{D(real_image)[0].cpu().detach().numpy()[0] }, '
              f'D_fake:{D(fake_image3)[0].cpu().detach().numpy()[0]}')

        # 将生成的假样本进行保存展示
        fake_show = np.zeros([8 * 28, 8 * 28], dtype=float)
        q = 0
        for i in range(8):
            for j in range(8):
                fake_show[i * 28:i * 28 + 28, j * 28:j * 28 + 28] = np.reshape(fake_image[q, :].data.cpu(), (28, 28))
                q += 1
        fake_show = mapnp(fake_show, 0, 255)  # 将fake_show反归一化到区间（0,255）

        # 存储生成假样本到指定文件夹中
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        plt.imshow(fake_show, cmap='gray')
        plt.savefig(f'{result_path}/fake_img{epoch + 1}.jpg')

        # 将生成网络和判别网络传回CPU，进行保存
        G_cpu = G.cpu()
        D_cpu = D.cpu()
        torch.save(G_cpu, 'G(PAPERLoss).pkl')
        torch.save(D_cpu, 'D(PAPERLoss).pkl')

        # 将生成器和判别器重新传回cpu
        G = G.cuda()
        D = D.cuda()
