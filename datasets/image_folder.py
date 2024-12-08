import os
import json
import random
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


class ImageFolder(Dataset):

#     def __init__(self, spec, split_file=None, split_key=None, first_k=None):
#         self.hr_path = spec.hr_path
#         self.lr_path = spec.lr_path
#         self.repeat = spec.repeat
#         self.augment = spec.augment

#         if split_file is None:
#             filenames = sorted(os.listdir(self.hr_path))
#         else:
#             with open(split_file, 'r') as f:
#                 filenames = json.load(f)[split_key]
#         if first_k is not None:
#             filenames = filenames[:first_k]

#         self.files = filenames

#     def __len__(self):
#         return len(self.files) * self.repeat

#     def __getitem__(self, idx):
#         filename = self.files[idx % len(self.files)]
#         hr_file = os.path.join(self.hr_path, filename)
#         lr_file = os.path.join(self.lr_path, filename)

#         if not (os.path.exists(hr_file) and os.path.exists(lr_file)):
#             raise FileNotFoundError(f"HR file {hr_file} or LR file {lr_file} not found.")

#         hr_image = Image.open(hr_file).convert('RGB')
#         lr_image = Image.open(lr_file).convert('RGB')

#         hr_image = transforms.ToTensor()(hr_image)
#         lr_image = transforms.ToTensor()(lr_image)

#         # print(hr_image)

#         if self.augment:
#             hflip = random.random() < 0.5
#             vflip = random.random() < 0.5
#             dflip = random.random() < 0.5

#             def augment(x):
#                 if hflip:
#                     x = x.flip(-2)
#                 if vflip:
#                     x = x.flip(-1)
#                 if dflip:
#                     x = x.transpose(-2, -1)
#                 return x
            
#             hr_image = augment(hr_image)
#             lr_image = augment(lr_image)

#         return {
#             'hr': hr_image,
#             'lr': lr_image
#         }

    def __init__(self, spec, split_file=None, split_key=None, first_k=None):
        self.repeat = spec.repeat
        root_path = spec.root_path

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)
            self.files.append(transforms.ToTensor()(
                Image.open(file).convert('RGB')))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]
        return x