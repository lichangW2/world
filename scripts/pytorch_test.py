# -*- coding:utf-8 -*-

##
## 可用base: reg.qiniu.com/avaprd/pytorch0.4.0-python3.6-conda3-nvvl:20180528
##
##

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        print("x:",x)

        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
params = list(net.parameters())
print(len(params))
print(params[0].size())
input = torch.randn(2, 1, 32, 32)
out = net(input)
#out.requires_grad
#out.requires_grad_()
#out.grad
net.zero_grad()
#out.backward(torch.randn(1, 10))
target = torch.arange(1, 21)
target = target.view(2, -1)
criterion = nn.MSELoss()


loss = criterion(out,target)
loss.backward()
print(out.requires_grad)
print(loss.requires_grad)
