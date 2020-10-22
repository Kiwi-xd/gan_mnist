# gan_mnist
My first Gan code for generating MNIST datasets

数据集存储在data文件夹的MNIST中，raw中存放着官网下载的数据集
mnist_processing.py是读取数据集的程序，其中的函数可以将官网格式的数据集加载为图片矩阵
train（BCELoss）.py文件是训练程序，采用BCELoss作为损失函数
train（PAPERLoss）.py文件也是训练程序，采用GAN论文原文的损失函数
几个pkl文件分别是训练好的网络参数，如需在GPU上使用需要重新加载进GPU中

version
python 3.7.1
pytorch 1.1.0
matplotlib 2.2.3

