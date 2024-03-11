import torch.optim
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from tqdm import tqdm

from model import *

Learning_Rate = 1e-2
EPOCH = 20
writer = SummaryWriter('logs')

train_data = torchvision.datasets.CIFAR10('CIFAR', train=True, transform=transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10('CIFAR', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

model = CIFARModel()

loss_CE = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=Learning_Rate)


for epoch in range(EPOCH):
    print('epoch{} begin:'.format(epoch))

    # train
    model.train()
    for step, data in tqdm(enumerate(train_loader)):
        imgs, targets = data
        out = model(imgs)
        loss = loss_CE(out, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            writer.add_scalar('train_loss', loss, step)
            print('epoch{} step{} loss:{}'.format(epoch, step, loss))

    # cal loss & acc on val
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for step, data in tqdm(enumerate(test_loader)):
            imgs, targets = data
            out = model(imgs)
            val_loss += loss_CE(out, targets).item()
            val_acc += (out.argmax(1) == targets).sum()/len(test_data)
    print('{}th epoch val_loss: {}, val_acc:{}'.format(epoch, val_loss, val_acc))
    writer.add_scalar('val_loss', val_loss, epoch)
    writer.add_scalar('val_acc', val_acc, epoch)

    # save model
    torch.save(model, 'savedmodel/CIFARmodel_epoch{}.pth'.format(epoch))

writer.close()