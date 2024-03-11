import PIL.Image
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

# -1. SummaryWriter(dir)
writer = SummaryWriter("logs")

# 0. tensorboard --logdir mylogdir [-- port myport] @ terminal

# 1. writer.add_scalar(title, y, x(step))
# eg. y = x^3 - 2x + 1
for i in range(100):
    writer.add_scalar("y = x^3 - 2x + 1", i ** 3 - 2 * i + 1, i)


# 2.  writer.add_image(
#         title,
#         y: img(torch.Tensor, numpy.ndarray, or string/blobname),
#         x:step,
#         [dataformats = 'HWC']
#     )

#                               .img
# PIL.Image.open(img_path)  ->  <class 'PIL.JpegImagePlugin.JpegImageFile'>
# np.array(jpeg)            ->  <class 'numpy.ndarray'> （H, W, Channel）
# cv2.imread(img_path)      ->  <class 'numpy.ndarray'>


# img_path = 'data/train/ants/0013035.jpg'  step 1
img_path = 'data/train/bees/16838648_415acd9e3f.jpg'
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)

writer.add_image('test', img_array, 2, dataformats='HWC')

writer.close()
