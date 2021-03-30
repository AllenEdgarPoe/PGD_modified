'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import argparse
from models.wide_resnet_cifar import *
from torch.autograd import Variable
from torchattack.attacks.pgd import *
import torch.nn as nn
from robustbench.utils import clean_accuracy


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


norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--bool', default=False, type=bool)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

args = parser.parse_args()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
best_epoch = 0
iteration = 0

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# checkpoint = torch.load('./st_1_56.pth')
# checkpoint = torch.load('./st_2_56.pth')
# checkpoint = torch.load('./st_3_56.pth')
# checkpoint = torch.load('./st_4_56.pth')
student = WideResNet(depth=28, widen_factor=2, num_classes=10) #  72.73
student = nn.DataParallel(student)
student = nn.Sequential(norm_layer, student)
student.cuda()
# attack_model = WideResNet(depth=28, widen_factor=2, num_classes=10)
# attack_model = nn.DataParallel(attack_model)
# checkpoint = torch.load('./savemodel/ResNet28_2_cifar10/best_checkpoint.pth')
# attack_model.load_state_dict(checkpoint['model'])
# attack_model = nn.Sequential(norm_layer, attack_model)
# student = attack_model.cuda()

torch.backends.cudnn.deterministic = True
cudnn.benchmark = True
criterion_cross = nn.CrossEntropyLoss()

optimizer = optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    global iteration
    print('\nEpoch: %d' % epoch)
    student.train()

    train_loss = 0
    correct = 0
    total = 0
    if epoch < 80:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.1

    if epoch >= 80 and epoch < 120:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.01

    if epoch >= 120:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    atk_train = PGD(student, eps=8 / 255, alpha=4 / 255, steps=16)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        iteration = iteration + 1
        optimizer.zero_grad()

        inputs, targets = Variable(inputs), Variable(targets)
        # atk = torchattacks.PGDL2(student, eps=0.5, alpha=0.2, steps=7)
        # atk_train = torchattacks.PGD(student, eps=8 / 255, alpha=4 / 255, steps=16)
        adversarial_images = atk_train(inputs, targets)
        student_outputs = student(inputs)
        student_outputs_adv = student(adversarial_images)
        # loss_clean = criterion_cross(student_outputs, targets)
        loss_clean_adv = criterion_cross(student_outputs_adv, targets)
        # loss = 0.5 * loss_clean + 0.5 * loss_clean_adv
        loss = loss_clean_adv
        # loss = F.cross_entropy(student_outputs[1], targets.detach())
        loss.backward()
        optimizer.step()

        train_loss += loss
        _, predicted = torch.max(student_outputs, 1)
        total += targets.size(0)
        correct += predicted.eq(targets).cpu().sum()
        if (iteration % 50) == 0:
            print('train accuracy : %.4f  loss: %.4f' % (float(correct)/float(total)*100, loss))

def test(epoch):
    print('\nEpoch: %d' % epoch)
    global best_acc
    global best_epoch
    global iteration
    # teacher.eval()
    student.eval()

    clean_acc = 0
    adv_acc = 0
    total = 0
    atk_test = PGD(student, eps=8 / 255, alpha=4 / 255, steps=16)
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        acc = clean_accuracy(student, inputs, targets)
        clean_acc+=acc

        adv_images = atk_test(inputs, targets)
        acc = clean_accuracy(student, adv_images, targets)
        adv_acc+=acc

        total += 1


    print('Clean Test Accuracy: %.4f' % (float(clean_acc) / float(total) * 100))
    print('Purturb Test Accuracy: %.4f' % (float(adv_acc) / float(total) * 100))

    acc = clean_acc + adv_acc
    if acc > best_acc:
        print('saving')
        torch.save(student, './result2/WRN28_2_PGD.pth')
        best_acc = acc
        best_epoch = epoch
    print('best_ epoch :  %.d | best Acc :  %.3f   |   factor - with KD' % (
        best_epoch, best_acc))


for epoch in range(start_epoch, start_epoch+160):
    train(epoch)
    test(epoch)