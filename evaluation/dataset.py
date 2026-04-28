import torchvision.transforms.functional as F
import numpy as np
import random
import os
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms


# -------------------------
# 基础 Transform
# -------------------------

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label = data['image'], data['label']

        image = F.resize(image, self.size, interpolation=InterpolationMode.BILINEAR)
        label = F.resize(label, self.size, interpolation=InterpolationMode.NEAREST)

        return {'image': image, 'label': label, 'name': data['name']}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            return {
                'image': F.hflip(data['image']),
                'label': F.hflip(data['label']),
                'name': data['name']
            }
        return data


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            return {
                'image': F.vflip(data['image']),
                'label': F.vflip(data['label']),
                'name': data['name']
            }
        return data


class ToTensor(object):
    def __call__(self, data):
        image, label = data['image'], data['label']
        return {
            'image': F.to_tensor(image),     # [0,1]
            'label': F.to_tensor(label),     # [0,1]
            'name': data['name']
        }


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        image = F.normalize(data['image'], self.mean, self.std)
        return {
            'image': image,
            'label': data['label'],
            'name': data['name']
        }


# -------------------------
# 主 Dataset（支持 image / image_rgb 分流）
# -------------------------

class FullDataset(Dataset):
    def __init__(self, image_root, gt_root, size, mode):
        self.images = sorted([
            os.path.join(image_root, f)
            for f in os.listdir(image_root)
            if f.lower().endswith(('.jpg', '.png'))
        ])
        self.gts = sorted([
            os.path.join(gt_root, f)
            for f in os.listdir(gt_root)
            if f.lower().endswith(('.jpg', '.png'))
        ])

        assert len(self.images) == len(self.gts), "Image/GT 数量不匹配"

        base_tf = [
            Resize((size, size)),
        ]

        if mode == 'train':
            base_tf += [
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
            ]

        self.base_transform = transforms.Compose(base_tf)
        self.to_tensor = ToTensor()
        self.normalize = Normalize()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.rgb_loader(self.images[idx])
        label = self.binary_loader(self.gts[idx])

        name = os.path.splitext(os.path.basename(self.images[idx]))[0]

        data = {'image': image, 'label': label, 'name': name}
        data = self.base_transform(data)

        # --- image_rgb：给 MedSAM2 ---
        data = self.to_tensor(data)
        image_rgb = data['image'].clone()   # [0,1]，不 Normalize

        # --- image：给 DINO / 原模型 ---
        image_norm = self.normalize(data)['image']

        # --- label：确保是二值 ---
        label = (data['label'] > 0.5).float()

        return {
            'image': image_norm,      # Normalize 后
            'image_rgb': image_rgb,   # 原始 RGB [0,1]
            'label': label,
            'name': name
        }

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')


# -------------------------
# TestDataset（同样分流，给 SAM2 正确输入）
# -------------------------

class TestDataset:
    def __init__(self, image_root, gt_root, size):
        self.images = sorted([
            os.path.join(image_root, f)
            for f in os.listdir(image_root)
            if f.lower().endswith(('.jpg', '.png'))
        ])
        self.gts = sorted([
            os.path.join(gt_root, f)
            for f in os.listdir(gt_root)
            if f.lower().endswith('.png')
        ])

        self.size = size
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        label = self.binary_loader(self.gts[self.index])

        name = os.path.basename(self.images[self.index])

        image = F.resize(image, (self.size, self.size), interpolation=InterpolationMode.BILINEAR)
        label = F.resize(label, (self.size, self.size), interpolation=InterpolationMode.NEAREST)

        image_tensor = F.to_tensor(image)              # [0,1]
        image_rgb = image_tensor.clone()
        image_norm = F.normalize(
            image_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        label = (F.to_tensor(label) > 0.5).float()

        self.index += 1

        return {
            'image': image_norm.unsqueeze(0),
            'image_rgb': image_rgb.unsqueeze(0),
            'label': label,
            'name': name
        }

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')
