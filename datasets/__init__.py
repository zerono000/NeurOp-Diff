import torch
import random
import numpy as np

from . import image_folder
from . import wrappers

from utils import make_coord
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.0005

    if config.data.rescaled:
        X = 2 * X - 1.0
    elif config.data.logit_transform:
        X = logit_transform(X)

    return X


def inverse_data_transform(config, X):
    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)


def resize_fn(img, size):
    return transforms.Resize([size, size], interpolation=InterpolationMode.BICUBIC, antialias=True)(img)


def feed_data(config, data):
        p = random.uniform(config.wrapper.scale_min, config.wrapper.scale_max)

        img_hr = data['hr']
        # img_lr = data['lr']
        w_lr = round(img_hr.shape[-1] / round(p))
        img_lr = resize_fn(img_hr, w_lr)
        # print(p)
        # print(img_lr.shape)

        if config.dataset.augment:
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

            img_lr = augment(img_lr)

        hr_coord = make_coord(img_hr.shape[-2:], flatten=False)
        cell = torch.tensor([2 / img_hr.shape[-2], 2 / img_hr.shape[-1]], dtype=torch.float32)
        hr_coord = hr_coord.repeat(img_hr.shape[0], 1, 1, 1)
        cell = cell.repeat(img_hr.shape[0], 1)
        
        return {
            'gt': img_hr,
            'inp': img_lr,
            'coord': hr_coord,
            'cell': cell,
        }
