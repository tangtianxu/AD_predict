
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()

        # 定义多层感知机的层
        self.fc1 = nn.Linear(input_dim, 64)  # 输入层到隐藏层1
        self.bn1 = nn.BatchNorm1d(64)
        # self.fc2 = nn.Linear(64, 128)  # 隐藏层1到隐藏层2

        self.fc3 = nn.Linear(64, 32)  # 隐藏层2到隐藏层3
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 16)  # 隐藏层3到隐藏层4
        self.bn4 = nn.BatchNorm1d(16)
        self.fc5 = nn.Linear(16, output_dim)  # 隐藏层4到输出层


    def forward(self, x):
        # 通过每一层并应用ReLU激活函数
        x = F.relu(self.bn1(self.fc1(x)))

        # x = F.relu(self.fc2(x))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)  # 输出层，不需要ReLU，因为输出是一个类别概率分布

        return F.softmax(x, dim=1)  # 使用Softmax进行多分类
