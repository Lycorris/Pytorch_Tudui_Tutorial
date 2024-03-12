import torch.optim
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from tqdm import tqdm
import time


# from model import *

# Method 1
def use_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()  # gpu
    return x


# Method 2
def use_cuda_2(x, device='cpu'):
    if torch.cuda.is_available():
        device = 'cuda'
    if torch.backends.mps.is_available():
        device = 'mps'
    device = torch.device(device)
    return x.to(device)

# Configuration
Learning_Rate = 1e-2
EPOCH = 20
writer = SummaryWriter('logs')

# dataset
train_data = torchvision.datasets.CIFAR10('CIFAR', train=True, transform=transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10('CIFAR', train=False, transform=transforms.ToTensor())

# dataloader
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)


# model definition
class CIFARModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_maxp = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.MaxPool2d(2)
        )
        self.flat = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.conv_maxp(x)
        x = self.flat(x)
        x = self.mlp(x)
        return x


# model init
model = use_cuda_2(CIFARModel())

# loss func
loss_CE = use_cuda_2(nn.CrossEntropyLoss())
# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=Learning_Rate)

start_time = time.time()
for epoch in range(EPOCH):
    print('epoch{} begin:'.format(epoch))

    # train
    model.train()
    for step, data in tqdm(enumerate(train_loader)):
        imgs, targets = data
        imgs, targets = use_cuda_2(imgs), use_cuda_2(targets)
        out = model(imgs)
        loss = loss_CE(out, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            writer.add_scalar('train_loss', loss, step)
            end_time = time.time()
            print('epoch{} step{} time:{}, loss:{}'.format(epoch, step, end_time - start_time,loss))
            start_time = end_time

    # cal loss & acc on val
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for step, data in tqdm(enumerate(test_loader)):
            imgs, targets = data
            imgs, targets = use_cuda_2(imgs), use_cuda_2(targets)
            out = model(imgs)
            val_loss += loss_CE(out, targets).item()
            val_acc += (out.argmax(1) == targets).sum() / len(test_data)
    print('{}th epoch val_loss: {}, val_acc:{}'.format(epoch, val_loss, val_acc))
    writer.add_scalar('val_loss', val_loss, epoch)
    writer.add_scalar('val_acc', val_acc, epoch)

    # save model
    torch.save(model, 'savedmodel/CIFARmodel_epoch{}.pth'.format(epoch))

writer.close()
