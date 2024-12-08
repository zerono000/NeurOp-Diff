import os
import argparse
import yaml
import sys
import torch
import numpy as np

from tqdm import tqdm

from utils import calc_psnr, calc_ssim
from utils import Averager
from utils import tensor2img, save_img

from runners.diffusion import get_current_visuals

from models.ema import EMAHelper
# from models.diffusion import Model

from runners.diffusion import sample_image
from runners.diffusion import make_data_loader

from datasets import data_transform, inverse_data_transform, feed_data

from models.diffusion import UNet


def parse_args_and_config():
    parser = argparse.ArgumentParser(description="Super-Resolution Diffusion Model Evaluation")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to produce samples from the model",
    )
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )

    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    
    # Set device
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def eval_psrn(config, args, model, test_loader):
    model.eval()
    avg_psnr = Averager()
    avg_loss = Averager()
    avg_ssim = Averager()
    idx = 0

    result_path = '{}/{}_{}'.format('result', 'temporary', 120_0)
    os.makedirs(result_path, exist_ok=True)

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Generating image samples for PSNR Validate"):
            idx += 1
            # data = feed_data(config.test_dataset, batch)
            gt = data['gt']
            lr = data['inp']
            cell = data['cell']
            hr_coord = data['coord']

            lr = data_transform(config, lr)
            gt = data_transform(config, gt)

            gt = gt.to(config.device)
            lr = lr.to(config.device)
            cell = cell.to(config.device)
            hr_coord = hr_coord.to(config.device)

            shape = gt.shape
            x_t = torch.randn(
                shape[0],
                shape[1],
                shape[2],
                shape[3],
                device=config.device,
            )
            
            sr = sample_image(config, args, x_t, model, lr , hr_coord, cell).to(config.device)
            visuals = get_current_visuals(sr, data)
            sr_img = tensor2img(visuals['SR'])  # uint8
            hr_img = tensor2img(visuals['GT'])
            save_img(sr_img, '{}/{}_sr.png'.format(result_path, idx))
            # sr = model(lr, hr_coord, cell)

            loss = (gt - sr).abs().sum(dim=(1, 2, 3)).mean(dim=0)
            avg_loss.add(loss)
            
            sr = inverse_data_transform(config, sr)
            gt = inverse_data_transform(config, gt)

            psnr = calc_psnr(gt, sr)
            ssim = calc_ssim(sr_img, hr_img)
            avg_psnr.add(psnr)
            avg_ssim.add(ssim)

    return avg_psnr.item(), avg_ssim.item(), avg_loss.item()


def main():
    args, config = parse_args_and_config()
    print(f"Starting evaluation with checkpoint: {args.model}")
    print(f"Using device: {config.device}")

    test_loader = make_data_loader(config.test_dataset, 0, 0, tag='test')

    # model = Model(config)
    model = UNet(config)
    model = model.to(config.device)

    try:
        checkpoint = torch.load(args.model, map_location=config.device, weights_only=True)
        model.load_state_dict(checkpoint[0], strict=True)

        if config.model.ema:
            ema_helper = EMAHelper(mu=config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(checkpoint[-1])
            ema_helper.ema(model)
        else:
            ema_helper = None
            
        psnr, ssim, loss = eval_psrn(config, args, model, test_loader)
        print(f"PSNR: {psnr:12.6f}, SSIM: {ssim:.4e}, Loss: {loss:12.6f}")
    
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise

    return 0

if __name__ == "__main__":
    sys.exit(main())