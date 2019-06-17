# -*- coding: utf-8 -*-

"""
宿題。
Author :
    Yuki Kumon
Last Update :
    2019-06-17
"""


from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn  # ネットワーク構築用
import torch.optim as optim  # 最適化関数
import torch.nn.functional as F  # ネットワーク用の様々な関数
import torch.utils.data  # データセット読み込み関連

import numpy as np
from PIL import Image

# from torchsummary import summary
from tensorboardX import SummaryWriter

# import matplotlib.pyplot as plt


TENSOR_BOARD_LOG_DIR = './tensorboard_log'
writer = SummaryWriter(TENSOR_BOARD_LOG_DIR)


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class MyDataset(Dataset):
    '''
    dataset class
    '''

    def __init__(self, dict, trans1=None):
        self.data = dict[b'data']
        self.label = dict[b'labels']
        self.trans1 = trans1

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data = self.data[idx]
        r = np.reshape(data[:1024], (32, 32))
        g = np.reshape(data[1024:2048], (32, 32))
        b = np.reshape(data[2048:], (32, 32))
        img = np.empty([32, 32, 3])
        img[:, :, 0] = r
        img[:, :, 1] = g
        img[:, :, 2] = b
        img = img.astype(int)
        img = Image.fromarray(np.uint8(img))

        label = self.label[idx]

        if self.trans1:
            image = self.trans1(img)

        return image, label


class CNN(nn.Module):
    '''
    畳み込みニューラルネット
    '''
    def __init__(self, input_number, dropout_ratio):
        # initialization of class
        super(CNN, self).__init__()

        # define the convolution layer of encoder
        self.conv1_1 = nn.Conv2d(input_number, 64, kernel_size=3, stride=1, padding=1)
        self.bachnorm1_1 = nn.BatchNorm2d(num_features=64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bachnorm1_2 = nn.BatchNorm2d(num_features=64)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bachnorm2_1 = nn.BatchNorm2d(num_features=128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bachnorm2_2 = nn.BatchNorm2d(num_features=128)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bachnorm3_1 = nn.BatchNorm2d(num_features=256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bachnorm3_2 = nn.BatchNorm2d(num_features=256)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bachnorm3_3 = nn.BatchNorm2d(num_features=256)

        self.fc1 = nn.Linear(256 * 4 * 4, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 10)

        self.dropout = nn.Dropout2d(p=dropout_ratio)

    def flatten(self, x):
        return x.view(x.size(0), -1)

    def forward(self, x):
        x1_1 = F.relu(self.bachnorm1_1(self.conv1_1(x)))
        x1_2 = F.relu(self.bachnorm1_2(self.conv1_2(x1_1)))
        x1p, id1 = F.max_pool2d(self.dropout(x1_2), kernel_size=2, stride=2, return_indices=True)

        x2_1 = F.relu(self.bachnorm2_1(self.conv2_1(x1p)))
        x2_2 = F.relu(self.bachnorm2_2(self.conv2_2(x2_1)))

        x2p, id2 = F.max_pool2d(self.dropout(x2_2), kernel_size=2, stride=2, return_indices=True)

        x3_1 = F.relu(self.bachnorm3_1(self.conv3_1(x2p)))
        x3_2 = F.relu(self.bachnorm3_2(self.conv3_2(x3_1)))
        x3_3 = F.relu(self.bachnorm3_3(self.conv3_3(x3_2)))

        x3p, id3 = F.max_pool2d(self.dropout(x3_3), kernel_size=2, stride=2, return_indices=True)

        x = self.flatten(x3p)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def set_trans():
    trans1 = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return trans1


class typechange():
    def __init__(self):
        pass

    def __call__(self, X):
        return torch.from_numpy(X)


def train(epoch, is_cuda, loader, model, optimizer, criterion):
    '''
    train function
    '''
    model.train()
    for batch_idx, (image, label) in enumerate(loader):
        # forwadr
        optimizer.zero_grad()
        if is_cuda:
            image = image.to('cuda')
        output = model(image)
        loss = criterion(output, label)
        # print(output)
        # backward
        loss.backward()
        optimizer.step()
        # print(output.data.max(1)[1])
        # print(label)
        """
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(image), len(loader.dataset),
            100. * batch_idx / len(loader), loss.item()))
        """
    writer.add_scalar('train_loss', loss.item(), epoch)
    print('train epoch ', str(epoch), '')


def test(is_cuda, loader, model, criterion):
    '''
    testing function
    '''
    # initialize
    test_loss = 0.0
    correct = 0.0
    model.eval()
    for (image, label) in loader:
        # Variable型への変換(統合されたので省略)
        # image, label = Variable(image.float(), volatile=True), Variable(label)
        if is_cuda:
            image = image.to('cuda')
        output = model(image)
        test_loss += criterion(output, label).item()  # sum up batch loss
        if is_cuda:
            image = image.to('cpu')
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(label.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / (len(test_loader.dataset))))


"""
def onehot_encode(label, class_num=10):
    result = np.zeros(class_num, dtype=int)
    result[label] = 1
    return result
"""


if __name__ == '__main__':
    is_cuda = False
    epoch_max = 50
    train_path = ['./cifar-10-batches-py/data_batch_1', './cifar-10-batches-py/data_batch_2', './cifar-10-batches-py/data_batch_3', './cifar-10-batches-py/data_batch_4', './cifar-10-batches-py/data_batch_5']
    test_path = './cifar-10-batches-py/test_batch'

    trans1 = set_trans()
    train_loader = []
    for i in train_path:
        train_loader.append(torch.utils.data.DataLoader(MyDataset(unpickle(i), trans1), batch_size=16, shuffle=True))
    test_loader = torch.utils.data.DataLoader(MyDataset(unpickle(test_path), trans1), batch_size=16, shuffle=True)

    model = CNN(3, 0.2)
    if is_cuda:
        model = model.to('cuda')
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.CrossEntropyLoss()
    if(is_cuda):
        criterion = criterion.to('cuda')

    # 試しに
    # train(1, is_cuda, train_loader[0], model, optimizer, criterion)

    for epoch in range(1, 1 + epoch_max):
        loader_num = np.random.choice([0, 1, 2, 3, 4])
        train(1, is_cuda, train_loader[loader_num], model, optimizer, criterion)
        if epoch % 10 == 0:
            test(is_cuda, test_loader, model, criterion)

    # save
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    },
        './train.tar'
    )

    """
    hoge = unpickle('./cifar-10-batches-py/test_batch')
    hogeset = MyDataset(hoge)
    # print(hogeset[10])
    model = CNN(3, 0.2)
    summary(model, input_size=(3, 32, 32))
    # print(onehot_encode(7))
    """
