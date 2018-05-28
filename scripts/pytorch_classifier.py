##
## 可用base: reg.qiniu.com/avaprd/pytorch0.4.0-python3.6-conda3-nvvl:20180528
##
##
import sys,os
import traceback
import numpy as np

import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

if sys.version_info[0]==2:
    import cPickle
else:
    import _pickle as cPickle

class Data(Dataset):
    data_path = "/disk1/workspace/LichangWang/pytorch/data/cifar-10-batches-py/"
    train_list = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    test_list = ["test_batch"]
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0, 5), (0.5, 0.5, 0.5))])

    train_dt = None
    train_label = None
    test_dt = None
    test_label = None

    def __init__(self, train=True):
        self.is_train = train
        self.data_gen()

    def unpickle(self, file):
        """ load CIFAR-10  """
        with open(file, 'rb') as fo:
            dict = cPickle.load(fo)
        return dict

    def stack_dt(self, dt_list):

        data = []
        label = []
        try:
            for bt in dt_list:
                dtset = self.unpickle(self.data_path + bt)
                data.append(dtset["data"])
                label = label + dtset["labels"]

        except Exception as _e:
            traceback.format_exc()
            print  "error: ", _e
            return None, None
        return data, label

    def data_gen(self):
        """return train_data, train_label, test_data, test_label"""
        if self.is_train:
            train_data, train_lb = self.stack_dt(self.train_list)
            if train_data is None or train_lb is None:
                raise Exception("get data error")
            self.train_dt = np.concatenate(train_data)
            self.train_dt.reshape((50000, 3, 32, 32))
            self.train_dt.transpose((0, 2, 3, 1))  # convert to HWC for transform
            self.train_label = train_lb
        else:
            test_data, test_lb = self.stack_dt(self.test_list)
            if test_data is None or test_lb is None:
                raise Exception("get data error")

            self.test_dt = np.concatenate(test_data)
            self.test_dt.reshape((10000, 3, 32, 32))
            self.test_dt.transpose((0, 2, 3, 1))
            self.test_label = test_lb

    def __len__(self):
        if self.is_train:
            return len(self.train_dt)
        else:
            return len(self.test_dt)

    def __getitem__(self, index):
        if self.is_train:
            dt, label = self.train_dt[index], self.train_label[index]
        else:
            dt, label = self.test_dt[index], self.test_label[index]

        dt = self.transform(dt)
        return dt, label


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":

    train_set = Data()
    test_set = Data(train=False)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=True, num_workers=2)

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = net(inputs)
            print "labels:", labels, "\n outputs: ", outputs
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d,%5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))

    print("Finished Training")
    torchvision.datasets.CIFAR10
