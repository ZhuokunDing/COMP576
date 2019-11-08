from scipy import misc
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import matplotlib as mp
import torch.optim as optim
from torch.utils.data import DataLoader
import torch

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs')

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=1)
        self.pool = nn.MaxPool2d(2, padding=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.conv1(x))
        self.conv1_act = x.detach().clone().cpu().numpy()
        x = self.pool(x)
        x = self.act(self.conv2(x))
        self.conv2_act = x.detach().clone().cpu().numpy()
        x = self.pool(x)
        x = x.view(-1, 7 * 7 * 64)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x
        
def load_data():

    ntrain = 1000 # per class
    ntest = 100 # per class
    nclass = 10 # number of classes
    imsize = 28
    nchannels = 1 


    Train = np.zeros((ntrain*nclass,nchannels,imsize,imsize))
    Test = np.zeros((ntest*nclass,nchannels,imsize,imsize))
    LTrain = np.zeros(ntrain*nclass)
    LTest = np.zeros(ntest*nclass)

    itrain = -1
    itest = -1
    for iclass in range(0, nclass):
        for isample in range(0, ntrain):
            path = './data/CIFAR10/Train/%d/Image%05d.png' % (iclass,isample)
            im = misc.imread(path); # 28 by 28
            im = im.astype(float)/255
            itrain += 1
            Train[itrain,0,:,:] = im
            LTrain[itrain] = iclass
        for isample in range(0, ntest):
            path = './data/CIFAR10/Test/%d/Image%05d.png' % (iclass,isample)
            im = misc.imread(path); # 28 by 28
            im = im.astype(float)/255
            itest += 1
            Test[itest,0,:,:] = im
            LTest[itest] = iclass 
    return Train, LTrain, Test, LTest

def accuracy(net, imgs, labels):
    correct = []
    with torch.no_grad():
        net.eval()
        ys = net(imgs)
        correct = ys.argmax(dim=1) == labels
    return correct.float().mean()

def add_all(writer, var, var_name, iter_n):
    writer.add_scalar(var_name + '_mean', var.mean(), iter_n)
    writer.add_scalar(var_name + '_max', var.max(), iter_n)
    writer.add_scalar(var_name + '_min', var.mean(), iter_n)
    writer.add_scalar(var_name + '_std', var.std(), iter_n)
    writer.add_histogram(var_name + '_hist', var, iter_n)

def train_net(Train, LTrain, Test, LTest, lr, momentum, report=20, writer=writer):
    ntrain = 1000 # per class
    ntest = 100 # per class
    nclass = 10 # number of classes
    imsize = 28
    nchannels = 1 
    batchsize = 100
    nsamples = ntrain * nclass

    net = LeNet()
    net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    running_loss = 0.0
    batch_xs = torch.tensor(np.zeros((batchsize, nchannels, imsize, imsize)), device='cuda', dtype=torch.float32)#setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
    batch_ys = torch.tensor(np.zeros(batchsize), device='cuda', dtype=torch.long)#setup as [batchsize, the how many classes] 
    Test = torch.tensor(Test, device='cuda', dtype=torch.float32)
    LTest = torch.tensor(LTest, device='cuda', dtype=torch.long)


    for epoch in range(10):
        for i in range(int(nsamples/batchsize)): # try a small iteration size once it works then continue
            perm = np.arange(nsamples)
            np.random.shuffle(perm)
            for j in range(batchsize):
                batch_xs[j,:,:,:] = torch.tensor(Train[perm[j],:,:,:])
                batch_ys[j] = LTrain[perm[j]]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(batch_xs)
            loss = criterion(outputs, batch_ys)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i+1) % report == 0:    # print every {report} iterations
                print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / report))
                writer.add_scalar('training_loss', running_loss / report, epoch * nsamples + (i+1)*batchsize)
                running_loss = 0.0

                train_accuracy = accuracy(net, batch_xs, batch_ys)
                print(f'train_accuracy:{train_accuracy:.3f}')
                writer.add_scalar('training_accuracy', train_accuracy, epoch * nsamples + (i+1)*batchsize)

                test_accuracy = accuracy(net, Test, LTest)
                print(f'test_accuracy:{test_accuracy:.3f}')
                writer.add_scalar('test_accuracy', test_accuracy, epoch * nsamples + (i+1)*batchsize)

                fig, axes = plt.subplots(4,8)
                for (weight,axe) in zip(net.conv1.weight, axes.ravel()):
                    axe.axis('off')
                    axe.imshow(weight.detach().squeeze().cpu().numpy())
                writer.add_figure('conv1_weight', fig, epoch * nsamples + (i+1)*batchsize)

                add_all(writer, net.conv1_act, 'conv1_activation', epoch * nsamples + (i+1)*batchsize)
                add_all(writer, net.conv2_act, 'conv2_activation', epoch * nsamples + (i+1)*batchsize)

    return net


    