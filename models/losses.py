import torch

from models.denoising import generalized_steps
from utils import calc_ssim
from datasets import inverse_data_transform


def noise_estimation_loss(model,
                          x_t: torch.Tensor,
                          inp: torch.Tensor,
                          hr_coord: torch.Tensor,
                          cell: torch.Tensor,
                          e: torch.Tensor,
                          b: torch.Tensor,
                          continuous_sqrt_alpha_cumprod,
                          keepdim=False):
    output = model(x_t, continuous_sqrt_alpha_cumprod, inp, hr_coord, cell)
    sum_pixel = e.shape[1]*e.shape[2]*e.shape[3]
    if keepdim:
        return (e - output).abs().sum(dim=(1, 2, 3))/sum_pixel
    else:
        return (e - output).abs().sum(dim=(1, 2, 3)).mean(dim=0)/sum_pixel


loss_registry = {
    'simple': noise_estimation_loss,
}
