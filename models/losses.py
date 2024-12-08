import torch


def noise_estimation_loss(model,
                          x_t: torch.Tensor,
                        #   x_0: torch.Tensor,
                        #   t: torch.LongTensor,
                          inp: torch.Tensor,
                          hr_coord: torch.Tensor,
                          cell: torch.Tensor,
                          e: torch.Tensor,
                          b: torch.Tensor,
                          continuous_sqrt_alpha_cumprod,
                          keepdim=False):
    # a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    # x_t = x_0 * a.sqrt() + e * (1.0 - a).sqrt()
    # output = model(inp, hr_coord, cell)
    output = model(x_t, continuous_sqrt_alpha_cumprod, inp, hr_coord, cell)
    # print(output.shape)
    # print(e.shape)
    # if keepdim:
    #     return (x_0 - output).square().sum(dim=(1, 2, 3))
    # else:
    #     return (x_0 - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
    if keepdim:
        return (e - output).abs().sum(dim=(1, 2, 3))
    else:
        return (e - output).abs().sum(dim=(1, 2, 3)).mean(dim=0)


loss_registry = {
    'simple': noise_estimation_loss,
}
