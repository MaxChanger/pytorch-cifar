# -*- coding:utf-8 -*-  
'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

# 定义是否使用GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 超参数设置
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
train_loss_everyepoch = 0
train_accuracy_everyepoch = 0
test_loss_everyepoch = 0
test_accuracy_everyepoch = 0


# Data
print('==> Preparing data..')
# torchvision.transforms是pytorch中的图像预处理包 一般用Compose把多个步骤整合到一起：
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),   # 先四周填充0，再把图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),      # 以0.5的概率水平翻转给定的PIL图像
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


## if you cannot run the program because of "OUT OF MEMORY", you can decrease the batch_size properly.
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

# DataLoader接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor
# 生成一个个batch进行批训练，组成batch的时候顺序打乱取
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

print('SAMPLES:', len(trainset), len(testset))
print('EPOCH:', len(trainloader), len(testloader))
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') # 类型

# Model
print('==> Building model..')

# 定义卷积神经网络
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
        # fully connect
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# net = Net()
## use model
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
net = ShuffleNetV2(1)

## use pretrained model
# import torchvision.models as models
# # net = models.resnet50(pretrained=True)
# net = models.vgg16(pretrained=True)
# net.fc = nn.Linear(2048, 10)

netname = 'ShuffleNetV2'
writer = SummaryWriter(comment='ShuffleNetV2_train') # 提供一个 comment 参数，将使用 runs/日期时间-comment 路径来保存日志
writer_total = SummaryWriter(comment='ShuffleNetV2_test')

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join('./checkpoint', netname, 'ckpt.t7'))
    #checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# 定义损失函数和优化器 
criterion = nn.CrossEntropyLoss()   # 损失函数为交叉熵，多用于多分类问题
# 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training 训练网络
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device) 
        optimizer.zero_grad()   # 梯度清零 zero the parameter gradients
        outputs = net(inputs)   # forward
        loss = criterion(outputs, targets)  # loss
        loss.backward()     # 执行反向传播 backward
        optimizer.step()    # 新参数 update weights
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    train_loss_everyepoch = train_loss/(batch_idx+1)
    train_accuracy_everyepoch = 100.*correct/total
    return (train_loss_everyepoch,train_accuracy_everyepoch)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        test_error_everyepoch = test_loss/(batch_idx+1)
        test_accuracy_everyepoch = 100.*correct/total

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if not os.path.isdir('checkpoint/'+netname):
            os.mkdir('checkpoint/'+netname)
        torch.save(state, './checkpoint/'+netname + '/ckpt.t7')
        best_acc = acc
    return (test_error_everyepoch,test_accuracy_everyepoch)

# 训练网络
for epoch in range(start_epoch, start_epoch+200):
    (train_loss_everyepoch,train_accuracy_everyepoch) = train(epoch)
    # if (epoch+1)%5 == 0: # 每5次epoch测试一次
    (test_loss_everyepoch,test_accuracy_everyepoch) = test(epoch)

    writer.add_scalar('loss', train_loss_everyepoch, global_step= epoch)
    writer.add_scalar('accuracy', train_accuracy_everyepoch, global_step= epoch)
    # writer.add_scalar('test_loss', test_loss_everyepoch, global_step= epoch)
    # writer.add_scalar('test_accuracy', test_accuracy_everyepoch, global_step= epoch)
    
    # writer_total.add_scalar('train_loss', train_loss_everyepoch, global_step= epoch)
    # writer_total.add_scalar('train_accuracy', train_accuracy_everyepoch, global_step= epoch)
    writer_total.add_scalar('loss', test_loss_everyepoch, global_step= epoch)
    writer_total.add_scalar('accuracy', test_accuracy_everyepoch, global_step= epoch)
    