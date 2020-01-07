# Must install pillow <= 6.1

import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
import torchvision.utils as v_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# Set Hyperparameters
# change num_gpu to the number of gpus you want to use

epoch = 100
batch_size = 512
learning_rate = 0.0002

# Download Data

mnist_train = dset.MNIST("./", train=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
                        ]),
                        target_transform=None,
                        download=True)

# Set Data Loader(input pipeline)

train_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=batch_size,shuffle=True,drop_last=True)


# Generator receives random noise z and create 1x28x28 image
# we can name each layer using OrderedDict

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(100, 7 * 7 * 256),
            nn.BatchNorm1d(7 * 7 * 256),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(OrderedDict([
            ('conv1', nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)),  # batch x 128 x 14 x 14
            ('bn1', nn.BatchNorm2d(128)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)),  # batch x 64 x 14 x 14
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU()),
        ]))
        self.layer3 = nn.Sequential(OrderedDict([
            ('conv3', nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1)),  # batch x 16 x 28 x 28
            ('bn3', nn.BatchNorm2d(16)),
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)),  # batch x 1 x 28 x 28
            ('relu4', nn.Tanh())
        ]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, z):
        out = self.layer1(z)
        out = out.view(batch_size, 256, 7, 7)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


# Discriminator receives 1x28x28 image and returns a float number 0~1
# we can name each layer using OrderedDict

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)),  # batch x 8 x 28 x 28
            # ('bn1',nn.BatchNorm2d(8)),
            ('relu1', nn.LeakyReLU()),
            ('conv2', nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)),  # batch x 16 x 14 x 14
            ('bn2', nn.BatchNorm2d(16)),
            ('relu2', nn.LeakyReLU()),
        ]))
        self.layer2 = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)),  # batch x 32 x 14 x 14
            ('bn3', nn.BatchNorm2d(32)),
            ('relu3', nn.LeakyReLU()),
            ('conv4', nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)),  # batch x 64 x 7 x 7
            ('bn4', nn.BatchNorm2d(64)),
            ('relu4', nn.LeakyReLU())
        ]))
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(batch_size, -1)
        out = self.fc(out)
        return out

# DataParallel 함수를 이용해 멀티 GPU에 모델을 올릴 수 있습니다.
# torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)
# device_ids: default all devices / output_device: default device 0

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

generator = nn.DataParallel(Generator()).to(device)
discriminator = nn.DataParallel(Discriminator()).to(device)

# Get parameter list by using class.state_dict().keys()

gen_params = generator.state_dict().keys()
dis_params = discriminator.state_dict().keys()

for i in gen_params:
    print(i)

# loss function, optimizers, and labels for training

loss_func = nn.MSELoss()
gen_optim = torch.optim.Adam(generator.parameters(), lr=learning_rate,betas=(0.5,0.999))
dis_optim = torch.optim.Adam(discriminator.parameters(), lr=learning_rate,betas=(0.5,0.999))

ones_label = torch.ones(batch_size,1).to(device)
zeros_label = torch.zeros(batch_size,1).to(device)

# model restore if any

import os

try:
  os.mkdir("./model")
except:
  pass

try:
  os.mkdir("./result")
except:
  pass

try:
    #generator, discriminator = torch.load('./model/dcgan.pkl')
    print("\n--------model restored--------\n")
except:
    print("\n--------model not restored--------\n")
    pass

# train

for i in range(epoch):
    for j, (image, label) in enumerate(train_loader):
        image = image.to(device)

        # generator
        gen_optim.zero_grad()

        z = init.normal_(torch.Tensor(batch_size, 100), mean=0, std=0.1).to(device)
        gen_fake = generator.forward(z)
        dis_fake = discriminator.forward(gen_fake)

        gen_loss = torch.sum(loss_func(dis_fake, ones_label))  # fake classified as real
        gen_loss.backward()
        gen_optim.step()

        # discriminator
        dis_optim.zero_grad()

        z = init.normal_(torch.Tensor(batch_size, 100), mean=0, std=0.1).to(device)
        gen_fake = generator.forward(z)
        dis_fake = discriminator.forward(gen_fake)

        dis_real = discriminator.forward(image)
        dis_loss = torch.sum(loss_func(dis_fake, zeros_label)) + torch.sum(loss_func(dis_real, ones_label))
        dis_loss.backward()
        dis_optim.step()

        # model save
        if j % 1000 == 0:
            torch.save([generator, discriminator], './model/dcgan.pkl')
            print("{}th iteration gen_loss: {} dis_loss: {}".format(i, gen_loss.data, dis_loss.data))
            v_utils.save_image(gen_fake.cpu().data[0:25], "./result/gen_{}_{}.png".format(i, j), nrow=5)

    # image_check(gen_fake.cpu())

from glob import glob

for i in range(0, epoch, 5):
    print(i)
    file_list = glob("./result/gen_{}_*.png".format(i))
    print(file_list)

    img_per_epoch = len(file_list)

    plt.figure(figsize=(25, 5))
    for idx, j in enumerate(file_list):
        img = plt.imread(j)
        plt.subplot(1, img_per_epoch, idx + 1)
        plt.imshow(img)
    plt.show()

