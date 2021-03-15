from models.resnet_cifar import *
from models.wide_resnet_cifar import *
from pgd_attack import *
from utils import *
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


class PGD:
    def __init__(self, args):
        self.args = args

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
        ### Data load
        if args.dataset == "cifar10":
            print('=> loading cifar10 data...')
            trainset = torchvision.datasets.CIFAR10(root=self.args.data_path, train=True, download=False,
                                                    transform=transform_train)
            self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True, num_workers=4)

            testset = torchvision.datasets.CIFAR10(root=self.args.data_path, train=False, download=False,
                                                   transform=transform_test)
            self.testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)


        #load model
        norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]) #normalize for cifar10
        if args.model == "resnet110":
            if args.dataset == "cifar100":
                self.model = nn.Sequential(norm_layer, resnet110_cifar(num_classes=100))
            elif args.dataset == "cifar10":
                # self.model = resnet110_cifar(num_classes=10)
                self.model = nn.Sequential(norm_layer, resnet110_cifar(num_classes=10))
            else:
                print("Dataset needs to be fixed")
                assert False
        elif args.model == "WRN":
            if args.dataset == "cifar100":
                self.model = nn.Sequential(norm_layer, WideResNet(depth=28, widen_factor=2, num_classes=10))
            elif args.dataset == "cifar10":
                self.model = nn.Sequential(norm_layer, WideResNet(depth=28, widen_factor=2, num_classes=10))
            else:
                print("Dataset needs to be fixed")
                assert False


        # self.model = torch.nn.DataParallel(self.model).cuda()
        self.model = self.model.cuda()
        torch.backends.cudnn.deterministic = True
        cudnn.benchmark = True

        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.epoch = 0

        self.save_path = args.save_path
        if args.train_attacker == "PGD":
            self.train_attacker = PGD_attack(self.model, self.args.epsilon, self.args.alpha, self.args.attack_steps, random_start=self.args.random_start)
        elif args.train_attacker == "PGD_mod":
            self.train_attacker = GD(self.model, self.args.epsilon, self.args.alpha, self.args.attack_steps, random_start=self.args.random_start)

        if args.test_attacker == "PGD":
            self.test_attacker = PGD_attack(self.model, self.args.epsilon, self.args.alpha, self.args.attack_steps, random_start=self.args.random_start)
        elif args.test_attacker == "PGD_mod":
            self.test_attacker = GD(self.model, self.args.epsilon, self.args.alpha, self.args.attack_steps, random_start=self.args.random_start)

        # resume from checkpoint
        # checkpoint_path = osp.join(args.save_path, 'checkpoint.pth')
        # if osp.exists(checkpoint_path):
        #     self._load_from_checkpoint(checkpoint_path)
        if not args.model_load == None:
            checkpoint_path = osp.join(args.model_load)
            self._load_from_checkpoint(checkpoint_path)

    def _log(self, message):
        print(message)
        f = open(osp.join(self.save_path, 'log.txt'), 'a+')
        f.write(message + '\n')
        f.close()

    def _load_from_checkpoint(self, checkpoint_path):
        print('Loading model from {} ...'.format(checkpoint_path))
        model_data = torch.load(checkpoint_path)
        self.model.load_state_dict(model_data['model'])
        self.optimizer.load_state_dict(model_data['optimizer'])
        self.epoch = model_data['epoch'] + 1
        print('Model loaded successfully')

    def _save_checkpoint(self, path):
        self.model.eval()
        model_data = dict()
        model_data['model'] = self.model.state_dict()
        model_data['optimizer'] = self.optimizer.state_dict()
        model_data['epoch'] = self.epoch
        torch.save(model_data, osp.join(self.save_path, path))

    def train(self):
        adv_losses = AverageMeter()
        nat_losses = AverageMeter()

        best_adv_acc = 0
        best_nat_acc = 0

        if self.epoch < 80:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.1

        if self.epoch >= 80 and self.epoch < 120:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.01

        if self.epoch >= 120:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.001


        while self.epoch < self.args.epochs:
            self.model.train()
            total = 0
            adv_correct = 0
            nat_correct = 0

            for i, (image,label) in enumerate(self.trainloader):
                if torch.cuda.is_available():
                    image = image.cuda()
                    label = label.cuda()

                self.optimizer.zero_grad()
                image, label = Variable(image), Variable(label)
                x_adv = self.train_attacker(image, label)

                #compute output
                adv_logits = self.model(x_adv)
                nat_logits = self.model(image)
                adv_loss = self.criterion(adv_logits, label)
                nat_loss = self.criterion(nat_logits, label)
                loss = 0.5*adv_loss+0.5*nat_loss
                # loss = adv_loss
                loss.backward()
                self.optimizer.step()

                #checking for attack-success acc
                _, adv_pred = torch.max(adv_logits, dim=1)
                adv_correct += (adv_pred == label).sum()
                total += label.size(0)

                #checking for natural-success acc
                _, nat_pred = torch.max(nat_logits, dim=1)
                nat_correct += (nat_pred == label).sum()

                adv_losses.update(adv_loss.data.item(), x_adv.size(0))
                nat_losses.update(nat_loss.data.item(), image.size(0))


            self.epoch += 1

            nat_acc = float(nat_correct)/total
            adv_acc = float(adv_correct)/total
            mess = "{}th Epoch, nat Acc: {:.3f}, adv Acc: {:.3f}, Loss: {:.3f}".format(self.epoch, nat_acc, adv_acc, loss.item())
            self._log(mess)
            self._save_checkpoint('checkpoint.pth')

            # Evaluation
            nat_acc, adv_acc = self.eval()

            self._log('Natural Accuracy: {:.3f}'.format(nat_acc))
            self._log('Adv Accuracy: {:.3f}'.format(adv_acc))

            if nat_acc + adv_acc > best_adv_acc + best_nat_acc:
                best_adv_acc = adv_acc
                best_nat_acc = nat_acc
                self._save_checkpoint('best_checkpoint.pth')
                self._log('Best Test Accuracy: {:.3f}/{:.3f}'.format(best_adv_acc, best_nat_acc))
        self._log('=======Best Test Accuracy: {:.3f}/{:.3f}======'.format(best_adv_acc, best_nat_acc))


    def eval(self):
        self.model.eval()
        adv_correct = 0
        nat_correct = 0
        total = 0

        for i, (image, label) in enumerate(self.testloader):
            if torch.cuda.is_available():
                image = image.cuda()
                label = label.cuda()
            image, label = Variable(image), Variable(label)
            adv_image = self.test_attacker(image, label)

            nat_logits = self.model(image)
            adv_logits = self.model(adv_image)

            _, nat_pred = torch.max(nat_logits, dim=1)
            _, adv_pred = torch.max(adv_logits, dim=1)

            nat_correct += (nat_pred == label).sum()
            adv_correct += (adv_pred == label).sum()
            total += label.size(0)

        nat_acc = float(nat_correct) / total
        adv_acc = float(adv_correct) / total
        return nat_acc, adv_acc





#
# class FixedTestloader(torch.utils.data.Dataset):
#   def __init__(self, testloader, test_attacker):
#       self.x_data = []
#       self.y_data = []
#       for i, (image, label) in enumerate(testloader):
#           x_adv = test_attacker.forward(image, label)
#           self.x_data.append(x_adv)
#           self.y_data.append(label)
#
#   def __len__(self):
#     return len(self.x_data)
#
#   def __getitem__(self, idx):
#     x = self.x_data[idx]
#     y = self.y_data[idx]
#     return x, y



