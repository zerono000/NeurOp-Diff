import os
import logging
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from tqdm import tqdm

from models.diffusion import Model
# from models.diffusion import UNet
from models.ema import EMAHelper
from models.optimizer import get_optimizer
from models.losses import loss_registry
from models.denoising import generalized_steps, ddpm_steps

from datasets.image_folder import ImageFolder
from datasets.wrappers import SRImplicitDownsampledFast
from datasets import data_transform, inverse_data_transform, feed_data

from utils import calc_psnr
from utils import Averager

from init_weight import init_weights

import torchvision.utils as tvu
from torch.utils.data import DataLoader

from scheduler import CustomLRScheduler

from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = ImageFolder(spec.dataset)
    # dataset = SRImplicitDownsampledFast(spec.wrapper, dataset)
    loader = DataLoader(dataset, batch_size=spec.batch_size,
        shuffle=(tag == 'train'), num_workers=10, pin_memory=True, persistent_workers=True)

    return loader


def make_data_loaders(config):
    train_loader = make_data_loader(config.train_dataset, tag='train')
    val_loader = make_data_loader(config.val_dataset, tag='val')

    return train_loader, val_loader

    
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


def sample_image(config, args, x, model, srno_input, coord, cell, last=True):

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

        xs = generalized_steps(x, seq, model, betas, srno_input, coord, cell, eta=args.eta)
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

        x = ddpm_steps(x, seq, model, betas, srno_input, coord, cell)
        
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
        # self.sqrt_alphas_cumprod_prev = np.sqrt(
        #     np.append(1., alphas_cumprod.cpu().numpy()))
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
        # model = UNet(config)
        # init_weights(model, init_type="orthogonal")
        # init_weights(model, init_type="kaiming")
        # init_weights(model, init_type="normal")
        model = model.to(self.device)

        optimizer = get_optimizer(self.config, model.parameters())
        # scheduler = CustomLRScheduler(optimizer, 0.00002, 0.00006, 4)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=6, T_mult=1, eta_min=0.000005)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        max_psnr = -1e18
        start_epoch, step = 1, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "epoch_best.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs+1):
            data_start = time.time()
            data_time = 0
            avg_loss = Averager()
            print(f"----------------- Executing training iteration {epoch}. -----------------")
            for i, data in enumerate(train_loader):
                data = feed_data(config.train_dataset, data)
                x = data['gt']
                inp = data['inp']
                cell = data['cell']
                hr_coord = data['coord']
                # print(x.shape)
                # print(inp.shape)
                # print(hr_coord.shape)
                # print(cell.shape)
                # print(x)

                # in_range = (x < 0.0)
                # exists_in_range = in_range.any().item()
                # logging.info(f"是否存在在范围内的值: {exists_in_range}")
                # print(inp)

                x = data_transform(self.config, x)
                inp = data_transform(self.config, inp)

                x = x.to(self.device)
                inp = inp.to(self.device)
                cell = cell.to(self.device)
                hr_coord = hr_coord.to(self.device)

                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                e = torch.randn_like(x).to(self.device)
                b = self.betas
                # t_ = np.random.randint(1, self.num_timesteps + 1)
                # continuous_sqrt_alpha_cumprod = torch.FloatTensor(
                #     np.random.uniform(
                #         self.sqrt_alphas_cumprod_prev[t_-1],
                #         self.sqrt_alphas_cumprod_prev[t_],
                #         size=n
                #     )
                # ).to(self.device)
                # continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(n, -1)

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                loss = loss_registry[config.model.type](model, x, t, inp, hr_coord, cell, e, b)

                avg_loss.add(loss)

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"Training... step: {step:7d},  loss: {loss.item():12.6f}"
                )
                # for param in model.parameters():
                #     if param.grad is not None: # and param.grad.norm() > 1000 
                #         logging.info(f"Gradient norm: {param.grad.norm()}")

                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         grad_norm = param.grad.norm().item()
                #         if grad_norm > 10000:
                #             logging.info({
                #                 'name': name,
                #                 'grad_norm': grad_norm,
                #                 'shape': param.shape
                #             })

                # for name, param in model.named_parameters():
                #     # if param.grad is not None:
                #     #     logging.info(f"{name} weight update: {param.grad.abs().mean()}")
                #     if torch.isnan(param).any():
                #         print(f"NaN found in {name}")

                # loss = loss / 2

                # avg_loss.add(loss)

                # if (i+1) % 2 == 0:
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
            
            logging.info(
                    f"Training... epoch: {epoch:2d} Loss: {avg_loss.item():12.6f}"
            )
            # save model
            # if epoch % self.config.training.epoch_save_freq == 0 or epoch == 1:
            states = [
                model.state_dict(),
                optimizer.state_dict(),
                epoch,
                step,
            ]
            if self.config.model.ema:
                states.append(ema_helper.state_dict())
                
            print("-------------------------- Begin validation ----------------------------")

            current_psnr, current_loss = self.validate(config, args, model, val_loader)
            logging.info(
                f"Validating... epoch: {epoch:2d} PSNR: {current_psnr:12.6f}"
            )
            logging.info(
                f"Validating... epoch: {epoch:2d} Loss: {current_loss:12.6f}"
            )

            print("------------------------ Model is being saved --------------------------")

            torch.save(states, os.path.join(self.args.log_path, 'epoch_{}_{:.2f}.pth'.format(epoch, current_psnr)))

            if current_psnr > max_psnr:
                max_psnr = current_psnr
                
                torch.save(states, os.path.join(self.args.log_path, 'epoch_best.pth'))

            print("--------------------------- End of saving ------------------------------")

            scheduler.step()

            logging.info(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
            # logging.info(f"Current learning rate: {scheduler.get_lr()[0]}")

    def validate(self, config, args, model, val_loader):        
        model.eval()
        avg_psnr = Averager()
        avg_loss = Averager()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Generating image samples for PSNR Validate"):
                data = feed_data(config.val_dataset, batch)
                gt = data['gt']
                lr = data['inp']
                cell = data['cell']
                hr_coord = data['coord']

                gt = data_transform(config, gt)
                lr = data_transform(config, lr)

                gt = gt.to(self.device)
                lr = lr.to(self.device)
                cell = cell.to(self.device)
                hr_coord = hr_coord.to(self.device)

                shape = gt.shape
                x_t = torch.randn(
                    shape[0],
                    shape[1],
                    shape[2],
                    shape[3],
                    device=self.device,
                )
                
                sr = sample_image(config, args, x_t, model, lr , hr_coord, cell).to(self.device)
                # print(sr.shape)
                # print(gt.shape)
                loss = (gt - sr).abs().sum(dim=(1, 2, 3)).mean(dim=0)
                print(loss)
                avg_loss.add(loss)
                
                gt = inverse_data_transform(config, gt)
                sr = inverse_data_transform(config, sr)

                psnr = calc_psnr(gt, sr)
                avg_psnr.add(psnr, shape[0])

        return avg_psnr.item(), avg_loss.item()

    # def sample(self, srno_input, coord, cell):
    #     model = Model(self.config)

    #     if not self.args.use_pretrained:
    #         if getattr(self.config.sampling, "ckpt_id", None) is None:
    #             states = torch.load(
    #                 os.path.join(self.args.log_path, "ckpt.pth"),
    #                 map_location=self.config.device,
    #             )
    #         else:
    #             states = torch.load(
    #                 os.path.join(
    #                     self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
    #                 ),
    #                 map_location=self.config.device,
    #             )
    #         model = model.to(self.device)
    #         model = torch.nn.DataParallel(model)
    #         model.load_state_dict(states[0], strict=True)

    #         if self.config.model.ema:
    #             ema_helper = EMAHelper(mu=self.config.model.ema_rate)
    #             ema_helper.register(model)
    #             ema_helper.load_state_dict(states[-1])
    #             ema_helper.ema(model)
    #         else:
    #             ema_helper = None
    #     else:
    #         # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
    #         if self.config.data.dataset == "CIFAR10":
    #             name = "cifar10"
    #         elif self.config.data.dataset == "LSUN":
    #             name = f"lsun_{self.config.data.category}"
    #         else:
    #             raise ValueError
    #         ckpt = get_ckpt_path(f"ema_{name}")
    #         print("Loading checkpoint {}".format(ckpt))
    #         model.load_state_dict(torch.load(ckpt, map_location=self.device))
    #         model.to(self.device)
    #         model = torch.nn.DataParallel(model)

    #     model.eval()

    #     if self.args.fid:
    #         self.sample_fid(model, srno_input, coord, cell)
    #     elif self.args.interpolation:
    #         self.sample_interpolation(model, srno_input, coord, cell)
    #     elif self.args.sequence:
    #         self.sample_sequence(model, srno_input, coord, cell)
    #     else:
    #         raise NotImplementedError("Sample procedeure not defined")

    # def sample_fid(self, model, srno_input, coord, cell):
    #     config = self.config
    #     img_id = len(glob.glob(f"{self.args.image_folder}/*"))
    #     print(f"starting from image {img_id}")
    #     total_n_samples = 50000
    #     n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

    #     with torch.no_grad():
    #         for _ in tqdm.tqdm(
    #             range(n_rounds), desc="Generating image samples for FID evaluation."
    #         ):
    #             n = config.sampling.batch_size
    #             x = torch.randn(
    #                 n,
    #                 config.data.channels,
    #                 config.data.image_size,
    #                 config.data.image_size,
    #                 device=self.device,
    #             )

    #             x = self.sample_image(x, model, srno_input, coord, cell)
    #             x = inverse_data_transform(config, x)

    #             for i in range(n):
    #                 tvu.save_image(
    #                     x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
    #                 )
    #                 img_id += 1

    # def sample_sequence(self, model, srno_input, coord, cell):
    #     config = self.config

    #     x = torch.randn(
    #         8,
    #         config.data.channels,
    #         config.data.image_size,
    #         config.data.image_size,
    #         device=self.device,
    #     )

    #     # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
    #     with torch.no_grad():
    #         _, x = self.sample_image(x, model, srno_input, coord, cell, last=False)

    #     x = [inverse_data_transform(config, y) for y in x]

    #     for i in range(len(x)):
    #         for j in range(x[i].size(0)):
    #             tvu.save_image(
    #                 x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
    #             )
