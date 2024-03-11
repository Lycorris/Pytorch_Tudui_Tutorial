from PIL.JpegImagePlugin import JpegImageFile
from torch.utils.data import Dataset
from PIL import Image
import os


class MyDataset(Dataset):

    def __init__(self, root_dir: str, label_dir: str):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.dir_path = os.path.join(root_dir, label_dir)
        self.img_name_list = os.listdir(self.dir_path)

    def __getitem__(self, idx: int) -> (JpegImageFile, str):
        img_item_name = self.img_name_list[idx]
        img_item_path = os.path.join(self.dir_path, img_item_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self) -> int:
        return len(self.img_name_list)


root_dir = '/Users/lycoris/Desktop/extend/learn_torch/data/train'
ants_label_dir = 'ants'
bees_label_dir = 'bees'
ants_dataset = MyDataset(root_dir, ants_label_dir)
bees_dataset = MyDataset(root_dir, bees_label_dir)

train_dataset = ants_dataset + bees_dataset


