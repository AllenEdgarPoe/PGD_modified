import argparse
import os
import time
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchattack.attacks.pgd import *
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from models.wide_resnet_cifar import *
from PIL import Image
from torch.utils.data import Sampler, RandomSampler
import numpy as np

best_prec = 0
best_adv_prec = 0

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std

class BatchSampler(RandomSampler):
    def __init__(self, sampler, batch_size, drop_last, multiscale_step=None, img_sizes=None):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        if multiscale_step is not None and multiscale_step < 1:
            raise ValueError("multiscale_step should be > 0, but got "
                             "multiscale_step={}".format(multiscale_step))
        if multiscale_step is not None and img_sizes is None:
            raise ValueError("img_sizes must a list, but got img_sizes={} ".format(img_sizes))

        self.multiscale_step = multiscale_step
        self.img_sizes = img_sizes

    def __iter__(self):
        num_batch = 0
        batch = []
        size = 32
        for idx in self.sampler:
            batch.append([idx, size])
            if len(batch) == self.batch_size:
                yield batch
                num_batch += 1
                batch = []
                if self.multiscale_step and num_batch % self.multiscale_step == 0:
                    size = np.random.choice(self.img_sizes)
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class MultiscaleDataSet(torchvision.datasets.ImageFolder):
    """Multiscale ImageFolder dataset"""

    def __getitem__(self, index):
        if isinstance(index, (tuple, list)):
            index, input_size = index
        else:
            # set the default image size here
            input_size = 32
        path, target = self.samples[index]
        sample = self.loader(path)

        # resize the image
        sample = sample.resize((input_size, input_size))

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # return the image and label
        return sample, target

def main():
    global best_prec
    global best_adv_prec

    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = WideResNet(depth=28, widen_factor=2, num_classes=200)
    model = nn.Sequential(norm_layer, model)
    model = nn.DataParallel(model).cuda()
    # model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), 0.1, momentum=0.9, weight_decay=5e-4)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True


    # normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    train_transform = transforms.Compose([
        # transforms.RandomCrop(8, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize,
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # normalize
    ])

    # train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform= train_transform)
    # trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=7)
    train_dataset = MultiscaleDataSet(root='./data/tiny-imagenet-200/train', transform=train_transform)
    train_batch_sampler = BatchSampler(sampler=RandomSampler(train_dataset), batch_size=512, multiscale_step=1, drop_last=True, img_sizes=[64,48,32])
    trainloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_sampler=train_batch_sampler, num_workers=7)

    test_dataset = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/val', transform=test_transform)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)


    for epoch in range(0, 160):
        if epoch < 80:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.1

        if epoch >= 80 and epoch < 120:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.01

        if epoch >= 120:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001


        # train for one epoch
        train(trainloader, model, criterion, optimizer, epoch)

        # evaluate on test set
        prec, adv_prec = validate(testloader, model, criterion)

        # remember best precision and save checkpoint
        if best_prec + best_adv_prec < prec + adv_prec:
            best_prec = prec
            best_adv_prec = adv_prec
            torch.save(model, './result2/WRN28_multi_at_tiny')

        print('Best nat Prec:{:.3f}, adv Prec:{:.3f}'.format(best_prec, best_adv_prec))


def train(trainloader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    adv_top1 = AverageMeter()
    model.train()

    atk_train = PGD(model, eps=8/255, alpha=4/255, steps=5)
    for i, (input, target) in enumerate(trainloader):
        input, target = input.cuda(), target.cuda()
        input, target = Variable(input), Variable(target)
        adv_input = atk_train(input, target)
        # compute output
        output = model(input)
        adv_output = model(adv_input)
        # loss = criterion(output, target)
        loss = criterion(adv_output, target)

        # measure accuracy and record loss
        prec = accuracy(output, target)[0]
        adv_prec = accuracy(adv_output, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))
        adv_top1.update(adv_prec.item(), adv_input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%100 == 0:
            # print('Epoch: {}, train accuracy : ({top1.avg:.3f})  loss: ({loss.avg:.3f})'.format(epoch, top1=top1, loss=losses))
            print('Epoch: {}, nat acc : ({top1.avg:.3f}), adv acc:  ({adv_top1.avg:.3f}),  loss: ({loss.avg:.3f})'.format(epoch, top1=top1, adv_top1=adv_top1, loss=losses))


def validate(val_loader, model, criterion):
    # switch to evaluate mode
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    adv_top1 = AverageMeter()

    atk_test = PGD(model, eps=8/255, alpha=4/255, steps=5)
    # with torch.no_grad():
    for i, (input, target) in enumerate(val_loader):
        input, target = input.cuda(), target.cuda()
        input, target = Variable(input), Variable(target)
        adv_input = atk_test(input, target)

        # compute output
        output = model(input)
        adv_output = model(adv_input)
        loss = criterion(adv_output, target)

        # measure accuracy
        prec = accuracy(output, target)[0]
        adv_prec = accuracy(adv_output, target)[0]

        losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))
        adv_top1.update(adv_prec.item(), adv_input.size(0))


    # print('test accuracy : ({top1.avg:.3f})  loss: ({loss.avg:.3f})'.format(top1=top1,loss=losses))
    print('nat acc : ({top1.avg:.3f}), adv acc:  ({adv_top1.avg:.3f}),  loss: ({loss.avg:.3f})'.format(top1=top1, adv_top1=adv_top1, loss=losses))

    return top1.avg, adv_top1.avg




def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
