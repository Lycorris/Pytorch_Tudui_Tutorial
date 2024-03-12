import torch
from PIL import Image
from torchvision import transforms

from model import *

img_path = 'data/dog.png'
img = Image.open(img_path)

trans = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])
img = trans(img)
img = torch.reshape(img, (1, 3, 32, 32))
img = img.to('mps')     # if model have params in 'mps'/'cuda' form, input need to transform
# map model to 'cpu'
# model = torch.load('savedmodel/CIFARmodel_epoch19.pth', map_location=torch.device('cpu'))


model = torch.load('savedmodel/CIFARmodel_epoch19.pth')
model.eval()
with torch.no_grad():
    out = model(img)

# {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
# 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
print(out)
print(out.argmax(1))
