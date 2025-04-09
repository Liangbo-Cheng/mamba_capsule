import os
from PIL import Image, ImageFile
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许加载截断的

class test_dataset:
    def __init__(self, image_root, gt_root):
        self.img_list = [os.path.splitext(f)[0] for f in os.listdir(gt_root) if f.endswith('.png')]
        self.image_root = image_root
        self.gt_root = gt_root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.img_list)
        self.index = 0

    def load_data(self):
        #image = self.rgb_loader(self.images[self.index])
        image = self.binary_loader(os.path.join(self.image_root,self.img_list[self.index]+ '.png'))
        gt = self.binary_loader(os.path.join(self.gt_root,self.img_list[self.index] + '.png'))
        self.index += 1
        return image, gt

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        # with open(path, 'rb') as f:
        #     img = Image.open(f)
        #     return img.convert('L')
        
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('L')
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return None  # 处理损坏的图片

