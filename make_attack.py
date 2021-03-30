import torchvision.transforms as transforms
from models.wide_resnet_cifar import *
from torchattack.attacks.pgd import *
from torchattack.attacks.pgdl2 import *
import torchvision
from attack import *

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
# checkpoint = torch.load('./savemodel/CIFAR10/WRN28_2_PGD_test.pth')
attack_model = WideResNet(depth=28, widen_factor=2, num_classes=10)
attack_model = nn.DataParallel(attack_model)
attack_model = nn.Sequential(norm_layer, attack_model)
# attack_model.load_state_dict(checkpoint.state_dict())

attack_model.cuda()

#load cifar10
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

atk1 = PGDL2(attack_model, eps=8 / 255, alpha=4 / 255, steps=5)
a = atk1.save(testloader, save_path='adv_examples/PGDL2/test_8_4_5.pth', verbose=True)
atk2 = PGD(attack_model, eps=8 / 255, alpha=4 / 255, steps=5)
b = atk2.save(testloader, save_path='adv_examples/PGD/test_8_4_5.pth', verbose=True)
print(a,b)