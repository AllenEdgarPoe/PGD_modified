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

class adv_KD:
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
        self.norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]) #normalize for cifar10
        if args.model == "resnet110":
            if args.dataset == "cifar100":
                self.model = nn.Sequential(self.norm_layer, resnet110_cifar(num_classes=100))
            elif args.dataset == "cifar10":
                # self.model = resnet110_cifar(num_classes=10)
                self.model = nn.Sequential(self.norm_layer, resnet110_cifar(num_classes=10))
            else:
                print("Dataset needs to be fixed")
                assert False
        elif args.model == "WRN":
            if args.dataset == "cifar100":
                self.model = nn.Sequential(self.norm_layer, WideResNet(depth=28, widen_factor=2, num_classes=10))
            elif args.dataset == "cifar10":
                self.model = nn.Sequential(self.norm_layer, WideResNet(depth=28, widen_factor=2, num_classes=10))
            else:
                print("Dataset needs to be fixed")
                assert False


        self.model = torch.nn.DataParallel(self.model).cuda()
        # self.model = self.model.cuda()

        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.epoch = 0
        self.save_path = args.save_path

        #load teacher models
        self.T_net1 = nn.Sequential(self.norm_layer, WideResNet(depth=28, widen_factor=2, num_classes=10))
        self.T_net1 = torch.nn.DataParallel(self.T_net1).cuda()
        self.T_net1 = self._load_from_checkpoint(args.t1_checkpoint_path, self.T_net1)
        self.T_net2 = nn.Sequential(self.norm_layer, WideResNet(depth=28, widen_factor=2, num_classes=10))
        self.T_net2 = torch.nn.DataParallel(self.T_net2).cuda()
        self.T_net2 = self._load_from_checkpoint(args.t2_checkpoint_path, self.T_net2)

        torch.backends.cudnn.deterministic = True
        cudnn.benchmark = True

        self.T_net1.eval()
        self.T_net2.eval()



    def distillation(self, y, teacher_scores, labels, T, alpha):
        soft_target = F.softmax(teacher_scores / T, dim=1)
        hard_target = labels
        logp = F.log_softmax(y / T, dim=1)
        loss_soft_target = -torch.mean(torch.sum(soft_target * logp, dim=1))
        loss_hard_target = nn.CrossEntropyLoss()(y, hard_target)
        loss = loss_soft_target * T * T + alpha * loss_hard_target
        return loss

    def _log(self, message):
        print(message)
        f = open(osp.join(self.save_path, 'log.txt'), 'a+')
        f.write(message + '\n')
        f.close()

    def _load_from_checkpoint(self, checkpoint_path, model):
        print('Loading model from {} ...'.format(checkpoint_path))
        model_data = torch.load(checkpoint_path)
        model.load_state_dict(model_data['model'])
        print('Model loaded successfully')
        return model

    def _save_checkpoint(self, path):
        self.model.eval()
        model_data = dict()
        model_data['model'] = self.model.state_dict()
        model_data['optimizer'] = self.optimizer.state_dict()
        model_data['epoch'] = self.epoch
        torch.save(model_data, osp.join(self.save_path, path))

    def train(self):
        best_acc_1 = 0
        best_acc_2 = 0
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
            adv_correct_1 = 0
            adv_correct_2 = 0
            nat_correct = 0

            for i, (image,label) in enumerate(self.trainloader):

                if torch.cuda.is_available():
                    image = image.cuda()
                    label = label.cuda()

                self.optimizer.zero_grad()
                image, label = Variable(image), Variable(label)

                #load attack method
                atk1 = torchattacks.PGD(self.model, eps=8 / 255, alpha=2 / 255, steps=4)
                atk2 = torchattacks.PGDL2(self.model, eps=0.5, alpha=0.1, steps=7)
                adversarial_images1 = atk1(image,label)
                adversarial_images2 = atk2(image,label)
                outputs1 = self.T_net1(image)
                outputs2 = self.T_net2(image)

                student_outputs_adv1 = self.model(adversarial_images1)
                student_outputs_adv2 = self.model(adversarial_images2)
                student_output = self.model(image)

                loss_adv1 = self.distillation(student_outputs_adv1, outputs1, label, 3, 0.9)  # learn both from pgd AND
                loss_adv2 = self.distillation(student_outputs_adv2, outputs2, label, 3, 0.9)
                loss = loss_adv1 + loss_adv2
                loss.backward()
                self.optimizer.step()

                #checking for attack-success acc
                _, adv_pred_1 = torch.max(student_outputs_adv1, dim=1)
                _, adv_pred_2 = torch.max(student_outputs_adv2, dim=1)
                _, pred = torch.max(student_output, dim=1)
                adv_correct_1 += (adv_pred_1 == label).sum()
                adv_correct_2 += (adv_pred_2 == label).sum()
                nat_correct += (pred == label).sum()
                total += label.size(0)

            self.epoch += 1

            acc_1 = float(adv_correct_1)/total
            acc_2 = float(adv_correct_2)/total
            nat_acc_ = float(nat_correct)/total
            mess = "{}th Epoch, nat_acc: {:.3f}, Acc_1: {:.3f}, Acc_2: {:.3f}, Loss: {:.3f}".format(self.epoch, nat_acc_, acc_1, acc_2, loss.item())
            self._log(mess)
            self._save_checkpoint('checkpoint.pth')

            # Evaluation
            nat_acc, adv_acc_1, adv_acc_2 = self.eval_()

            if nat_acc + adv_acc_1 + adv_acc_2 > best_nat_acc + best_acc_1 + best_acc_2:
                best_nat_acc = nat_acc
                best_acc_1 = adv_acc_1
                best_acc_2 = adv_acc_2
                self._save_checkpoint('best_checkpoint.pth')
                self._log('Best Test Accuracy: {:.3f}/{:.3f}/{:.3f}'.format(best_nat_acc, best_acc_1, best_acc_2))
        self._log('=======Best Test Accuracy: {:.3f}/{:.3f}/{:.3f}======'.format(best_nat_acc, best_acc_1, best_acc_2))


    def eval_(self):
        self.model.eval()
        adv_correct_1 = 0
        adv_correct_2 = 0
        nat_correct = 0
        total = 0

        for i, (image, label) in enumerate(self.testloader):
            if torch.cuda.is_available():
                image = image.cuda()
                label = label.cuda()
            image, label = Variable(image), Variable(label)
            atk1 = torchattacks.PGD(self.model, eps=8 / 255, alpha=2 / 255, steps=4)
            atk2 = torchattacks.PGDL2(self.model, eps=0.5, alpha=0.1, steps=7)
            adversarial_images1 = atk1(image, label)
            adversarial_images2 = atk2(image, label)
            student_outputs = self.model(image)
            student_outputs_adv1 = self.model(adversarial_images1)
            student_outputs_adv2 = self.model(adversarial_images2)

            # checking for attack-success acc
            _, predicted = torch.max(student_outputs.data, 1)
            _, adv_pred_1 = torch.max(student_outputs_adv1, dim=1)
            _, adv_pred_2 = torch.max(student_outputs_adv2, dim=1)
            nat_correct += (predicted == label).sum()
            adv_correct_1 += (adv_pred_1 == label).sum()
            adv_correct_2 += (adv_pred_2 == label).sum()
            total += label.size(0)

        nat_acc = float(nat_correct) / total
        adv_acc_1 = float(adv_correct_1) / total
        adv_acc_2 = float(adv_correct_2) / total

        self._log('Natural Accuracy: {:.3f}'.format(nat_acc))
        self._log('Adv Accuracy_1: {:.3f}'.format(adv_acc_1))
        self._log('Adv Accuracy_2: {:.3f}'.format(adv_acc_2))
        return nat_acc, adv_acc_1, adv_acc_2


    def test(self):
        norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
        net1 = nn.Sequential(norm_layer, WideResNet(depth=28, widen_factor=2, num_classes=10))
        net1 = net1.cuda()
        checkpoint_path1 = osp.join(self.args.model_load1)
        model_data1 = torch.load(checkpoint_path1)
        net1.load_state_dict(model_data1['model'])

        net2 = nn.Sequential(self.norm_layer, WideResNet(depth=28, widen_factor=2, num_classes=10))
        net2 = net2.cuda()
        checkpoint_path2 = osp.join(self.args.model_load2)
        model_data2 = torch.load(checkpoint_path2)
        net2.load_state_dict(model_data2['model'])

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

        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)

            #Clean image
            output1 = net1(inputs)
            output2 = net2(inputs)

            # net1: PGD trained, net2: GD trained
            atk_pgd1 = PGD_attack(net1, self.args.epsilon, self.args.alpha, self.args.attack_steps, random_start=self.args.random_start)  #noise
            atk_pgd2 = PGD_attack(net2, self.args.epsilon, self.args.alpha, self.args.attack_steps, random_start=self.args.random_start)
            atk_gd1 = GD(net1, self.args.epsilon, self.args.alpha, self.args.attack_steps, random_start=self.args.random_start)
            atk_gd2 = GD(net2, self.args.epsilon, self.args.alpha, self.args.attack_steps, random_start=self.args.random_start)

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
        print('Noise From Net1 with  GD: Net1: %.2f Net2: %.2f\n' % (
        float(correct3) / float(total) * 100, float(correct4) / float(total) * 100))
        print('Noise From Net2 with PGD: Net1: %.2f Net2: %.2f' % (
        float(correct5) / float(total) * 100, float(correct6) / float(total) * 100))
        print('Noise From Net2 with  GD: Net1: %.2f Net2: %.2f\n' % (
        float(correct7) / float(total) * 100, float(correct8) / float(total) * 100))



# Clean Accuracy Net1: 88.49
# Clean Accuracy Net2: 89.39
#
# Noise From Net1 with PGD: Net1: 72.79 Net2: 65.77
# Noise From Net1 with  GD: Net1: 24.21 Net2: 70.93
#
# Noise From Net2 with PGD: Net1: 78.60 Net2: 50.11
# Noise From Net2 with  GD: Net1: 23.54 Net2: 72.97





