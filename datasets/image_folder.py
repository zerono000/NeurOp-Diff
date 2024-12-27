import os
import json
import random
import torch
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from utils import make_coord
from torchvision.transforms import InterpolationMode

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img)))

class ImageFolder(Dataset):

    def __init__(self, spec, image_path=None, split_file=None, split_key=None, first_k=None):
        self.repeat = spec.repeat
        self.augment = spec.augment
        self.image_path = image_path
        self.p = random.randint(spec.scale_min, spec.scale_max)

        if image_path == None:
            self.root_path = spec.root_path
        else:
            self.root_path = image_path

        if image_path:
            self.files = [image_path]
        else:
            if split_file is None:
                filenames = sorted(os.listdir(self.root_path))
            else:
                with open(split_file, 'r') as f:
                    filenames = json.load(f)[split_key]
            if first_k is not None:
                filenames = filenames[:first_k]

            self.files = filenames

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        filename = self.files[idx % len(self.files)]

        if self.image_path == None:
            hr_file = os.path.join(self.root_path, filename)
        else:
            hr_file = filename

        hr_image = Image.open(hr_file).convert('RGB')
        hr_image = transforms.ToTensor()(hr_image)
        w_lr = round(hr_image.shape[-1] / round(self.p))
        lr_image = resize_fn(hr_image, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x
            
            hr_image = augment(hr_image)
            lr_image = augment(lr_image)

        hr_coord = make_coord([hr_image.shape[-2], hr_image.shape[-1]], flatten=False)
        cell = torch.tensor([2 / hr_image.shape[-2], 2 / hr_image.shape[-1]], dtype=torch.float32)

        return {
            'gt': hr_image,
            'inp': lr_image,
            'coord': hr_coord,
            'cell': cell
        }
