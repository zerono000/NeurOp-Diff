import os
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from models.diffusion import Model
from models.ema import EMAHelper
from models.optimizer import get_optimizer
from models.losses import loss_registry
from models.denoising import generalized_steps, ddpm_steps

from tqdm import tqdm
from init_weight import init_weights
from datasets.image_folder import ImageFolder
from datasets.wrappers import SRImplicitDownsampledFast
from datasets import data_transform, inverse_data_transform

from utils import calc_psnr, calc_ssim
from utils import Averager
from utils import tensor2img, save_img

from collections import OrderedDict
from torch.utils.data import DataLoader, DistributedSampler
from scheduler import CustomIterationScheduler



def make_data_loader(spec, image_path=None, tag=''):
    if spec is None:
        return None

    dataset = ImageFolder(spec.dataset, image_path)
    loader = DataLoader(dataset, batch_size=spec.batch_size,
        shuffle=(tag == 'train'), num_workers=5, pin_memory=True, persistent_workers=True)

    return loader


def make_data_loaders(config):
    train_loader = make_data_loader(config.train_dataset, tag='train')
    val_loader = make_data_loader(config.val_dataset, tag='val')

    return train_loader, val_loader


def get_current_visuals(SR, data, need_LR=True, sample=False):
        out_dict = OrderedDict()
       
        out_dict['SR'] = SR.detach().float().cpu()
        out_dict['GT'] = data['gt'].detach().float().cpu()
        return out_dict

    
def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def sample_image(config, args, x, model, inp, coord, cell, last=True):

    try:
        skip = args.skip
    except Exception:
        skip = 1

    betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
    betas = torch.from_numpy(betas).float().to(config.device)
    num_timesteps = betas.shape[0]
    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(dim=0)
    sqrt_alphas_cumprod_prev = np.sqrt(
        np.append(1., alphas_cumprod.cpu().numpy()))

    if args.sample_type == "generalized":
        if args.skip_type == "uniform":
            skip = num_timesteps // args.timesteps
            seq = range(0, num_timesteps, skip)
        elif args.skip_type == "quad":
            seq = (
                np.linspace(
                    0, np.sqrt(num_timesteps * 0.8), args.timesteps
                )
                ** 2
            )
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError

        xs = generalized_steps(x, seq, model, betas, inp, coord, cell, sqrt_alphas_cumprod_prev, eta=args.eta)
        x = xs

    elif args.sample_type == "ddpm_noisy":
        if args.skip_type == "uniform":
            skip = num_timesteps // args.timesteps
            seq = range(0, num_timesteps, skip)
        elif args.skip_type == "quad":
            seq = (
                np.linspace(
                    0, np.sqrt(num_timesteps * 0.8), args.timesteps
                )
                ** 2
            )
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError

        x = ddpm_steps(x, seq, model, betas, inp, coord, cell)
        
    else:
        raise NotImplementedError
    if last:
        x = x[0][-1]
    return x


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod.cpu().numpy()))
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        
        train_loader, val_loader = make_data_loaders(config)

        model = Model(config)
        # init_weights(model.Unet, init_type="orthogonal")
        model = model.to(self.device)

        lr_sequence = [float(lr) for lr in config.scheduler.lr_sequence]
        step_size = config.scheduler.step_size

        optimizer = get_optimizer(self.config, model.parameters())
        scheduler = CustomIterationScheduler(optimizer, lr_sequence, step_size)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        max_psnr = -1e18
        start_epoch, step = 1, 1
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "step_best.pth"), weights_only=True)
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]+1
            step = states[3]+1
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs+1):
            data_start = time.time()
            data_time = 0
            avg_loss = Averager()

            print(f"----------------- Executing training iteration {epoch} epoch. -----------------")
            # for i, data in enumerate(train_loader):
            for data in tqdm(train_loader, desc="Training"):
                x = data['gt'].to(self.device)
                inp = data['inp'].to(self.device)
                cell = data['cell'].to(self.device)
                hr_coord = data['coord'].to(self.device)

                x = data_transform(self.config, x)
                inp = data_transform(self.config, inp)

                n = x.size(0)
                data_time += time.time() - data_start
                model.train()

                e = torch.randn_like(x).to(self.device)
                b = self.betas
                t_ = np.random.randint(1, self.num_timesteps + 1)
                continuous_sqrt_alpha_cumprod = torch.FloatTensor(
                    np.random.uniform(
                        self.sqrt_alphas_cumprod_prev[t_-1],
                        self.sqrt_alphas_cumprod_prev[t_],
                        size=n
                    )
                ).to(self.device)
                continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(n, -1)
                x_t = continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1) * x + (1 - continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1)**2).sqrt() * e

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                loss = loss_registry[config.model.type](model, x_t, inp, hr_coord, cell, e, b, continuous_sqrt_alpha_cumprod)
                avg_loss.add(loss)
                tb_logger.add_scalar("loss", loss, global_step=step)
                # print(f"Training... step: {step:7d},  pix_loss: {loss.item():10.7f}")
                
                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                data_start = time.time()

                # save model
                if (step % 20000 == 0):
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())
                        
                    print("-------------------------- Begin validation ----------------------------")

                    current_psnr, ssim = self.validate(config, args, model, val_loader, epoch)
                    logging.info(
                        f"Validating... epoch: {epoch:2d} PSNR: {current_psnr:12.6f}, SSIM: {ssim:.4e}"
                    )

                    print("------------------------ Model is being saved --------------------------")

                    torch.save(states, os.path.join(self.args.log_path, 'step_{}.pth'.format(step)))
                    if current_psnr > max_psnr:
                        max_psnr = current_psnr
                        torch.save(states, os.path.join(self.args.log_path, 'step_best.pth'))

                    print("--------------------------- End of saving ------------------------------")

                    logging.info(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

                scheduler.step(step)
                step += 1

            logging.info(f"Training... epoch: {epoch:2d} Loss: {avg_loss.item():12.6f}")

    def validate(self, config, args, model, val_loader, epoch):        
        model.eval()
        avg_psnr = Averager()
        avg_loss = Averager()
        avg_ssim = Averager()
        idx = 0

        result_path = '{}/{}'.format('result', epoch)
        os.makedirs(result_path, exist_ok=True)

        with torch.no_grad():
            for data in tqdm(val_loader, desc="Generating image samples for PSNR Validate"):
                idx += 1
                gt = data['gt'].to(self.device)
                lr = data['inp'].to(self.device)
                cell = data['cell'].to(self.device)
                hr_coord = data['coord'].to(self.device)

                gt = data_transform(config, gt)
                lr = data_transform(config, lr)

                x_t = torch.randn_like(gt, device=self.device)
                sr = sample_image(config, args, x_t, model, lr , hr_coord, cell).to(self.device)
                visuals = get_current_visuals(sr, data)
                sr_img = tensor2img(visuals['SR'])  # uint8
                hr_img = tensor2img(visuals['GT'])
                save_img(sr_img, '{}/{}_sr.png'.format(result_path, idx))
                
                gt = inverse_data_transform(config, gt)
                sr = inverse_data_transform(config, sr)

                psnr = calc_psnr(gt, sr)
                ssim = calc_ssim(sr_img, hr_img)
                avg_psnr.add(psnr)
                avg_ssim.add(ssim)

        return avg_psnr.item(), avg_ssim.item()
