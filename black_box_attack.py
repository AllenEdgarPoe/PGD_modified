from models.resnet_cifar import *
from models.wide_resnet_cifar import *
from pgd_attack import *
from utils import *
import copy
import torch
import torchvision
import torchvision.transforms as transforms
# from models.resnet_cifar import *
# from models.wide_resnet_cifar import *
from torchvision import models
import torch.optim as optim
import os.path as osp
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
import torchattacks

#load data
transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False,
                                        transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False,
                                       transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)


#load model
norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
net1 = nn.Sequential(norm_layer, WideResNet(depth=28, widen_factor=2, num_classes=10))
# net1 = resnet32_cifar(num_classes=10)
net1 = torch.nn.DataParallel(net1).cuda()
checkpoint_path1 = osp.join("./results/resnet32_cifar10/best_checkpoint.pth")
model_data1 = torch.load(checkpoint_path1)
net1.load_state_dict(model_data1['model'])

net2 = nn.Sequential(norm_layer, WideResNet(depth=28, widen_factor=2, num_classes=10))
net2 = torch.nn.DataParallel(net2).cuda()
checkpoint_path2 = osp.join("./results/pgd_train/WRN/cifar10/PGD_PGD/best_checkpoint.pth")
model_data2 = torch.load(checkpoint_path2)
net2.load_state_dict(model_data2['model'])

cudnn.deterministic = True
cudnn.benchmark = True

print("safely loaded models")
net1.eval()
net2.eval()

correct1 = 0
correct2 = 0
correct3 = 0
correct4 = 0
correct5 = 0
correct6 = 0
correct7 = 0
correct8 = 0
correct13 = 0
correct14 = 0

total = 0

for batch_idx, (inputs, targets) in enumerate(testloader):
    inputs, targets = inputs.cuda(), targets.cuda()
    inputs, targets = Variable(inputs), Variable(targets)

    #Clean image
    output1 = net1(inputs)
    output2 = net2(inputs)

    # net1: PGD trained, net2: GD trained
    atk_pgd1 = torchattacks.PGD(net1, eps=8 / 255, alpha=2 / 255, steps=4)  #noise
    atk_pgd2 = torchattacks.PGD(net2, eps=8 / 255, alpha=2 / 255, steps=4)
    atk_gd1 = torchattacks.PGDL2(net1, eps=0.5, alpha=0.1, steps=7)
    atk_gd2 = torchattacks.PGDL2(net2, eps=0.5, alpha=0.1, steps=7)

    adversarial_images_pgd1 = atk_pgd1(inputs, targets)
    adversarial_images_gd1 = atk_gd1(inputs, targets)
    adversarial_images_pgd2 = atk_pgd2(inputs, targets)
    adversarial_images_gd2 = atk_gd2(inputs, targets)
    net1_output_pgd1, net2_output_pgd1 = net1(adversarial_images_pgd1), net2(adversarial_images_pgd1)
    net1_output_gd1, net2_output_gd1 = net1(adversarial_images_gd1), net2(adversarial_images_gd1)
    net1_output_pgd2, net2_output_pgd2 = net1(adversarial_images_pgd2), net2(adversarial_images_pgd2)
    net1_output_gd2, net2_output_gd2 = net1(adversarial_images_gd2), net2(adversarial_images_gd2)

    _, predicted1 = torch.max(net1_output_pgd1.data, 1)
    _, predicted2 = torch.max(net2_output_pgd1.data, 1)
    _, predicted3 = torch.max(net1_output_gd1.data, 1)
    _, predicted4 = torch.max(net2_output_gd1.data, 1)

    _, predicted5 = torch.max(net1_output_pgd2.data, 1)
    _, predicted6 = torch.max(net2_output_pgd2.data, 1)
    _, predicted7 = torch.max(net1_output_gd2.data, 1)
    _, predicted8 = torch.max(net2_output_gd2.data, 1)

    _, predicted_clean1 = torch.max(output1.data, 1)
    _, predicted_clean2 = torch.max(output2.data, 1)

    # _, predicted = torch.max(student_outputs[1].data, 1)
    total += targets.size(0)
    correct1 += predicted1.eq(targets.data).cpu().sum()
    correct2 += predicted2.eq(targets.data).cpu().sum()
    correct3 += predicted3.eq(targets.data).cpu().sum()
    correct4 += predicted4.eq(targets.data).cpu().sum()
    correct5 += predicted5.eq(targets.data).cpu().sum()
    correct6 += predicted6.eq(targets.data).cpu().sum()
    correct7 += predicted7.eq(targets.data).cpu().sum()
    correct8 += predicted8.eq(targets.data).cpu().sum()
    correct13 += predicted_clean1.eq(targets.data).cpu().sum()
    correct14 += predicted_clean2.eq(targets.data).cpu().sum()

print('Clean Accuracy Net1: %.2f' % (float(correct13) / float(total) * 100))
print('Clean Accuracy Net2: %.2f\n' % (float(correct14) / float(total) * 100))
print('Noise From Net1 with PGD: Net1: %.2f Net2: %.2f' % (
float(correct1) / float(total) * 100, float(correct2) / float(total) * 100))
print('Noise From Net1 with  PGDL2: Net1: %.2f Net2: %.2f\n' % (
float(correct3) / float(total) * 100, float(correct4) / float(total) * 100))
print('Noise From Net2 with PGD: Net1: %.2f Net2: %.2f' % (
float(correct5) / float(total) * 100, float(correct6) / float(total) * 100))
print('Noise From Net2 with  PGDL2: Net1: %.2f Net2: %.2f\n' % (
float(correct7) / float(total) * 100, float(correct8) / float(total) * 100))