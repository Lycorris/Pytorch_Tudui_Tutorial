import torch
import torchvision
from torch import nn

# load pretrain model
vgg16 = torchvision.models.vgg16()

# add layer
vgg16.classifier.add_module('add_linear', nn.Linear(1000, 10))

# revise layer
vgg16.classifier[6] = nn.Linear(4096, 10)

# save model&param
torch.save(vgg16, 'vgg16_model_with_param.pth')
vgg16_read = torch.load('vgg16_model_with_param.pth')

# save param only
torch.save(vgg16.state_dict(), 'vgg16_param.pth')
vgg16_model = torchvision.models.vgg16()
vgg16_model.load_state_dict(torch.load('vgg16_param.pth'))

