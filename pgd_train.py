from models.resnet_cifar import *
from models.wide_resnet_cifar import *
from pgd_attack import *
from utils import *
import torch
import torchvision
import torchvision.transforms as transforms
from models.resnet_cifar import *
from models.wide_resnet_cifar import *
import torch.optim as optim
import os.path as osp
import torch.backends.cudnn as cudnn
import torch.nn.functional as F


class PGD_train:
    def __init__(self, args):
        self.args = args

        ### Data load
        if args.dataset == "cifar10":
            print('=> loading cifar10 data...')
            normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

            train_dataset = torchvision.datasets.CIFAR10(
                root=args.data_path,
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
            self.trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                           num_workers=2)

            test_dataset = torchvision.datasets.CIFAR10(
                root=args.data_path,
                train=False,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]))
            self.testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
        elif args.dataset == "cifar100":
            print('=> loading cifar100 data...')
            normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

            train_dataset = torchvision.datasets.CIFAR100(
                root=args.data_path,
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
            self.trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                           num_workers=2)

            test_dataset = torchvision.datasets.CIFAR100(
                root=args.data_path,
                train=False,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]))
            self.testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

        #load model
        if args.model == "resnet110":
            if args.dataset == "cifar100":
                self.model = resnet110_cifar(num_classes=100)
            elif args.dataset == "cifar10":
                self.model = resnet110_cifar(num_classes=10)
            else:
                print("Dataset needs to be fixed")
                assert False
        elif args.model == "WRN":
            if args.dataset == "cifar100":
                self.model = self.model = wide_resnet_cifar(depth=28, width=10, num_classes=100)
            elif args.dataset == "cifar10":
                self.model = resnet110_cifar(num_classes=10)
            else:
                print("Dataset needs to be fixed")
                assert False
        # self.model = torch.nn.DataParallel(self.model).cuda()
        self.model = self.model.cuda()
        self.optimizer = optim.SGD(self.model.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[60, 120, 160], gamma=0.2)
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.epoch = 0

        self.save_path = args.save_path
        if args.train_attacker == "PGD":
            self.train_attacker = PGD_attack(self.model, self.args.epsilon, self.args.alpha, self.args.attack_steps, random_start=self.args.random_start)
        elif args.train_attacker == "PGD_mod":
            self.train_attacker = PGD_attack_modified(self.model, self.args.epsilon, self.args.alpha, self.args.attack_steps, random_start=self.args.random_start)

        if args.test_attacker == "PGD":
            self.test_attacker = PGD_attack(self.model, self.args.epsilon, self.args.alpha, self.args.attack_steps, random_start=self.args.random_start)
        elif args.test_attacker == "PGD_mod":
            self.test_attacker = PGD_attack_modified(self.model, self.args.epsilon, self.args.alpha, self.args.attack_steps, random_start=self.args.random_start)

        # resume from checkpoint
        # checkpoint_path = osp.join(args.save_path, 'checkpoint.pth')
        # if osp.exists(checkpoint_path):
        #     self._load_from_checkpoint(checkpoint_path)
        # checkpoint_path = osp.join('./results/pgd_train/resnet110/PGD_mod_PGD_mod/checkpoint.pth')
        # self._load_from_checkpoint(checkpoint_path)

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
        self.lr_scheduler.load_state_dict(model_data['lr_scheduler'])
        self.epoch = model_data['epoch'] + 1
        print('Model loaded successfully')

    def _save_checkpoint(self):
        self.model.eval()
        model_data = dict()
        model_data['model'] = self.model.state_dict()
        model_data['optimizer'] = self.optimizer.state_dict()
        model_data['lr_scheduler'] = self.lr_scheduler.state_dict()
        model_data['epoch'] = self.epoch
        torch.save(model_data, osp.join(self.save_path, 'checkpoint.pth'))

    def train(self):
        adv_losses = AverageMeter()
        nat_losses = AverageMeter()

        best = 0

        while self.epoch < self.args.epochs:
            self.model.train()
            total = 0
            adv_correct = 0
            nat_correct = 0

            for i, (image,label) in enumerate(self.trainloader):
                if torch.cuda.is_available():
                    image = image.cuda()
                    label = label.cuda()

                x_adv = self.train_attacker.forward(image, label)

                #compute output
                self.optimizer.zero_grad()
                adv_logits = self.model(x_adv)
                nat_logits = self.model(image)
                adv_loss = self.criterion(adv_logits, label)
                nat_loss = self.criterion(nat_logits, label)
                loss = 0.5*adv_loss+0.5*nat_loss
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
            self.lr_scheduler.step()

            nat_acc = float(nat_correct)/total
            adv_acc = float(adv_correct)/total
            mess = "{}th Epoch, nat Acc: {:.3f}, adv Acc: {:.3f}, Loss: {:.3f}".format(self.epoch, nat_acc, adv_acc, loss.item())
            self._log(mess)
            # self._save_checkpoint()

            # Evaluation
            nat_acc = self.eval_nat()
            adv_acc = self.eval_adv()
            self._log('Natural Accuracy: {:.3f}'.format(nat_acc))
            self._log('Adv Accuracy: {:.3f}'.format(adv_acc))

            if nat_acc + adv_acc > best:
                best = nat_acc + adv_acc
                self._save_checkpoint()

    def eval_nat(self):
        self.model.eval()

        correct = 0
        total = 0

        for i, (image, label) in enumerate(self.testloader):
            if torch.cuda.is_available():
                image = image.cuda()
                label = label.cuda()

            with torch.no_grad():
                logits = self.model(image)
            _, pred = torch.max(logits, dim=1)
            correct += (pred == label).sum()
            total += label.size(0)

        acc = float(correct) / total
        return acc

    def eval_adv(self):
        self.model.eval()

        correct = 0
        total = 0

        for i, (image, label) in enumerate(self.testloader):
            if torch.cuda.is_available():
                image = image.cuda()
                label = label.cuda()

            x_adv = self.test_attacker.forward(image, label)
            with torch.no_grad():
                logits = self.model(x_adv)

            _, pred = torch.max(logits, dim=1)
            correct += (pred == label).sum()
            total += label.size(0)

        acc = float(correct) / total
        return acc




