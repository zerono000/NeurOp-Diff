# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from models.sronet import SRNO


# def get_timestep_embedding(timesteps, embedding_dim):
#     """
#     This matches the implementation in Denoising Diffusion Probabilistic Models:
#     From Fairseq.
#     Build sinusoidal embeddings.
#     This matches the implementation in tensor2tensor, but differs slightly
#     from the description in Section 3.5 of "Attention Is All You Need".
#     """
#     assert len(timesteps.shape) == 1

#     half_dim = embedding_dim // 2
#     emb = math.log(10000) / (half_dim - 1)
#     emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
#     emb = emb.to(device=timesteps.device)
#     emb = timesteps.float()[:, None] * emb[None, :]
#     emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
#     if embedding_dim % 2 == 1:  # zero pad
#         emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
#     return emb


# def nonlinearity(x):
#     # swish
#     return x*torch.sigmoid(x)


# def Normalize(in_channels):
#     return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    

# # class AdaptiveWeight(nn.Module):
# #     def __init__(self):
# #         super(AdaptiveWeight, self).__init__()
        
# #         self.weight_scale = nn.Parameter(torch.tensor(1.0))
# #         self.weight_shift = nn.Parameter(torch.tensor(0.0))
# #         self.weight_curve = nn.Parameter(torch.tensor(1.0))
        
# #     def forward(self, t, total_steps):
# #         t_norm = t / (total_steps - 1)
        
# #         base_weight = torch.maximum(t_norm, torch.tensor(0.2, device=t_norm.device))
        
# #         adapted_weight = self.weight_scale * (base_weight ** self.weight_curve) + self.weight_shift
        
# #         adapted_weight = torch.clamp(adapted_weight, 0.0, 1.0)
        
# #         return adapted_weight


# # def weight_decay(steps, total_steps, begin_weight=1.0, final_weight=0.1):
# #     decay_rate = -torch.log(torch.tensor(final_weight / begin_weight)) / (total_steps - 1)
# #     return begin_weight * torch.exp(-decay_rate * steps)

# # def weight_decay(steps, total_steps, begin_weight=0.1, final_weight=1.0):
# #     growth_rate = torch.log(torch.tensor(final_weight / begin_weight)) / (total_steps - 1)
# #     return begin_weight * torch.exp(growth_rate * steps)


# class Upsample(nn.Module):
#     def __init__(self, in_channels, with_conv):
#         super().__init__()
#         self.with_conv = with_conv
#         if self.with_conv:
#             self.conv = torch.nn.Conv2d(in_channels,
#                                         in_channels,
#                                         kernel_size=3,
#                                         stride=1,
#                                         padding=1)

#     def forward(self, x):
#         x = torch.nn.functional.interpolate(
#             x, scale_factor=2.0, mode="nearest")
#         if self.with_conv:
#             x = self.conv(x)
#         return x


# class Downsample(nn.Module):
#     def __init__(self, in_channels, with_conv):
#         super().__init__()
#         self.with_conv = with_conv
#         if self.with_conv:
#             # no asymmetric padding in torch conv, must do it ourselves
#             self.conv = torch.nn.Conv2d(in_channels,
#                                         in_channels,
#                                         kernel_size=3,
#                                         stride=2,
#                                         padding=0)

#     def forward(self, x):
#         if self.with_conv:
#             pad = (0, 1, 0, 1)
#             x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
#             x = self.conv(x)
#         else:
#             x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
#         return x


# class ResnetBlock(nn.Module):
#     def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
#                  dropout, temb_channels=512):
#         super().__init__()
#         self.in_channels = in_channels
#         out_channels = in_channels if out_channels is None else out_channels
#         self.out_channels = out_channels
#         self.use_conv_shortcut = conv_shortcut

#         self.norm1 = Normalize(in_channels)
#         self.conv1 = torch.nn.Conv2d(in_channels,
#                                      out_channels,
#                                      kernel_size=3,
#                                      stride=1,
#                                      padding=1)
#         self.temb_proj = torch.nn.Linear(temb_channels,
#                                          out_channels)
#         self.norm2 = Normalize(out_channels)
#         self.dropout = torch.nn.Dropout(dropout)
#         self.conv2 = torch.nn.Conv2d(out_channels,
#                                      out_channels,
#                                      kernel_size=3,
#                                      stride=1,
#                                      padding=1)
#         if self.in_channels != self.out_channels:
#             if self.use_conv_shortcut:
#                 self.conv_shortcut = torch.nn.Conv2d(in_channels,
#                                                      out_channels,
#                                                      kernel_size=3,
#                                                      stride=1,
#                                                      padding=1)
#             else:
#                 self.nin_shortcut = torch.nn.Conv2d(in_channels,
#                                                     out_channels,
#                                                     kernel_size=1,
#                                                     stride=1,
#                                                     padding=0)

#     def forward(self, x, temb):
#         h = x
#         h = self.norm1(h)
#         h = nonlinearity(h)
#         h = self.conv1(h)

#         h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

#         h = self.norm2(h)
#         h = nonlinearity(h)
#         h = self.dropout(h)
#         h = self.conv2(h)

#         if self.in_channels != self.out_channels:
#             if self.use_conv_shortcut:
#                 x = self.conv_shortcut(x)
#             else:
#                 x = self.nin_shortcut(x)

#         return x+h


# class AttnBlock(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.in_channels = in_channels

#         self.norm = Normalize(in_channels)
#         self.q = torch.nn.Conv2d(in_channels,
#                                  in_channels,
#                                  kernel_size=1,
#                                  stride=1,
#                                  padding=0)
#         self.k = torch.nn.Conv2d(in_channels,
#                                  in_channels,
#                                  kernel_size=1,
#                                  stride=1,
#                                  padding=0)
#         self.v = torch.nn.Conv2d(in_channels,
#                                  in_channels,
#                                  kernel_size=1,
#                                  stride=1,
#                                  padding=0)
#         self.proj_out = torch.nn.Conv2d(in_channels,
#                                         in_channels,
#                                         kernel_size=1,
#                                         stride=1,
#                                         padding=0)

#     def forward(self, x):
#         h_ = x
#         h_ = self.norm(h_)
#         q = self.q(h_)
#         k = self.k(h_)
#         v = self.v(h_)

#         # compute attention
#         b, c, h, w = q.shape
#         q = q.reshape(b, c, h*w)
#         q = q.permute(0, 2, 1)   # b,hw,c
#         k = k.reshape(b, c, h*w)  # b,c,hw
#         w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
#         w_ = w_ * (int(c)**(-0.5))
#         w_ = torch.nn.functional.softmax(w_, dim=2)

#         # attend to values
#         v = v.reshape(b, c, h*w)
#         w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
#         # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
#         h_ = torch.bmm(v, w_)
#         h_ = h_.reshape(b, c, h, w)

#         h_ = self.proj_out(h_)

#         return x+h_


# class Model(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
#         num_res_blocks = config.model.num_res_blocks
#         attn_resolutions = config.model.attn_resolutions
#         dropout = config.model.dropout
#         in_channels = config.model.in_channels
#         resolution = config.data.image_size
#         resamp_with_conv = config.model.resamp_with_conv
#         num_timesteps = config.diffusion.num_diffusion_timesteps
        
#         if config.model.type == 'bayesian':
#             self.logvar = nn.Parameter(torch.zeros(num_timesteps))
        
#         self.ch = ch
#         self.temb_ch = self.ch*4
#         self.num_resolutions = len(ch_mult)
#         self.num_res_blocks = num_res_blocks
#         self.resolution = resolution
#         self.in_channels = in_channels

#         self.num_timesteps = num_timesteps
        
#         # self.pre_conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
#         # self.pre_conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         # self.srno_ch = config.model.srno.width
#         self.srno = SRNO(config)
#         states = torch.load('/home/aiseon/Downloads/SRNO/save/_train_edsr-sronet/epoch-best.pth')
#         self.srno.load_state_dict(states['model'])
#         self.srno.eval()

#         #timestep embedding
#         self.temb = nn.Module()
#         self.temb.dense = nn.ModuleList([
#             torch.nn.Linear(self.ch,
#                             self.temb_ch),
#             torch.nn.Linear(self.temb_ch,
#                             self.temb_ch),
#         ])

#         # downsampling
#         # self.pre_conv3 = torch.nn.Conv2d(ch + self.srno_ch, ch, kernel_size=1)
#         self.conv_in = torch.nn.Conv2d(in_channels,
#                                        self.ch,
#                                        kernel_size=3,
#                                        stride=1,
#                                        padding=1)

#         curr_res = resolution
#         in_ch_mult = (1,)+ch_mult
#         self.down = nn.ModuleList()
#         block_in = None
#         for i_level in range(self.num_resolutions):
#             block = nn.ModuleList()
#             attn = nn.ModuleList()
#             block_in = ch*in_ch_mult[i_level]
#             block_out = ch*ch_mult[i_level]
#             for i_block in range(self.num_res_blocks):
#                 block.append(ResnetBlock(in_channels=block_in,
#                                          out_channels=block_out,
#                                          temb_channels=self.temb_ch,
#                                          dropout=dropout))
#                 block_in = block_out
#                 if curr_res in attn_resolutions:
#                     attn.append(AttnBlock(block_in))
#             down = nn.Module()
#             down.block = block
#             down.attn = attn
#             if i_level != self.num_resolutions-1:
#                 down.downsample = Downsample(block_in, resamp_with_conv)
#                 curr_res = curr_res // 2
#             self.down.append(down)

#         # middle
#         self.mid = nn.Module()
#         self.mid.block_1 = ResnetBlock(in_channels=block_in,
#                                        out_channels=block_in,
#                                        temb_channels=self.temb_ch,
#                                        dropout=dropout)
#         self.mid.attn_1 = AttnBlock(block_in)
#         self.mid.block_2 = ResnetBlock(in_channels=block_in,
#                                        out_channels=block_in,
#                                        temb_channels=self.temb_ch,
#                                        dropout=dropout)

#         # upsampling
#         self.up = nn.ModuleList()
#         for i_level in reversed(range(self.num_resolutions)):
#             block = nn.ModuleList()
#             attn = nn.ModuleList()
#             block_out = ch*ch_mult[i_level]
#             skip_in = ch*ch_mult[i_level]
#             for i_block in range(self.num_res_blocks+1):
#                 if i_block == self.num_res_blocks:
#                     skip_in = ch*in_ch_mult[i_level]
#                 block.append(ResnetBlock(in_channels=block_in+skip_in,
#                                          out_channels=block_out,
#                                          temb_channels=self.temb_ch,
#                                          dropout=dropout))
#                 block_in = block_out
#                 if curr_res in attn_resolutions:
#                     attn.append(AttnBlock(block_in))
#             up = nn.Module()
#             up.block = block
#             up.attn = attn
#             if i_level != 0:
#                 up.upsample = Upsample(block_in, resamp_with_conv)
#                 curr_res = curr_res * 2
#             self.up.insert(0, up)  # prepend to get consistent order

#         # end
#         self.norm_out = Normalize(block_in)
#         # self.norm_out1 = Normalize(ch*2)
#         self.conv_out = torch.nn.Conv2d(block_in,
#                                         out_ch,
#                                         kernel_size=3,
#                                         stride=1,
#                                         padding=1)

#         # self.output1 = torch.nn.Conv2d(out_ch,
#         #                               ch,
#         #                               kernel_size=3,
#         #                               stride=1,
#         #                               padding=1)

#         # # self.output2 = torch.nn.Conv2d(ch*2,
#         # #                               ch,
#         # #                               kernel_size=3,
#         # #                               stride=1,
#         # #                               padding=1)

#         # self.output3 = torch.nn.Conv2d(ch,
#         #                               out_ch,
#         #                               kernel_size=3,
#         #                               stride=1,
#         #                               padding=1)
    
#     def forward(self, x, t, inp, coord, cell):   
#         with torch.no_grad():
#             srno_feat = self.srno(inp, coord, cell)
#         srno_feat = srno_feat.detach()
            
#         # SRNO
#         # srno_feat = self.srno(inp, coord, cell)
#         # print(srno_feat.shape)
#         # srno_weights = weight_decay(t, self.num_timesteps)
#         # srno_weights = self.AdaptiveWeight(t, self.num_timesteps)
#         # srno_feat = srno_feat * srno_weights.view(-1, 1, 1, 1)
        
#         # e = x
#         # x = self.pre_conv1(x)
#         # x_ = x
#         # x = self.pre_conv2(x)
        
#         x = torch.cat([x, srno_feat], dim=1)
#         # # x = self.pre_conv3(x)

#         # timestep embedding
#         temb = get_timestep_embedding(t, self.ch)
#         temb = self.temb.dense[0](temb)
#         temb = nonlinearity(temb)
#         temb = self.temb.dense[1](temb)

#         # downsampling
#         hs = [self.conv_in(x)]
#         for i_level in range(self.num_resolutions):
#             for i_block in range(self.num_res_blocks):
#                 h = self.down[i_level].block[i_block](hs[-1], temb)
#                 if len(self.down[i_level].attn) > 0:
#                     h = self.down[i_level].attn[i_block](h)
#                 hs.append(h)
#             if i_level != self.num_resolutions-1:
#                 hs.append(self.down[i_level].downsample(hs[-1]))

#         # middle
#         h = hs[-1]
#         h = self.mid.block_1(h, temb)
#         h = self.mid.attn_1(h)
#         h = self.mid.block_2(h, temb)

#         # upsampling
#         for i_level in reversed(range(self.num_resolutions)):
#             for i_block in range(self.num_res_blocks+1):
#                 h = self.up[i_level].block[i_block](
#                     torch.cat([h, hs.pop()], dim=1), temb)
#                 if len(self.up[i_level].attn) > 0:
#                     h = self.up[i_level].attn[i_block](h)
#             if i_level != 0:
#                 h = self.up[i_level].upsample(h)

#         # end
#         h = self.norm_out(h)
#         h = nonlinearity(h)
#         h = self.conv_out(h)

#         # h = self.output1(h)
#         # h = self.norm_out1(h)
#         # h = F.gelu(h)

#         # h = self.output1(h)
#         # # h = F.gelu(h)

#         # h = self.output3(h)
#         return h


import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction

from models.sronet import SRNO


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x

states = torch.load('/home/aiseon/Downloads/save2/_train_edsr-sronet/epoch-best.pth', weights_only=True)

class UNet(nn.Module):
    def __init__(
        self,
        config,
        in_channel=6,
        out_channel=3,
        inner_channel=64,
        norm_groups=32,
        channel_mults=(1, 2, 4, 4, 8),
        attn_res=[16,],
        res_blocks=2,
        dropout=0,
        with_noise_level_emb=True,
        image_size=256,
    ):
        super().__init__()

        self.srno = SRNO(config)
        self.srno.load_state_dict(states['model'])
        self.srno.eval()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                        dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x, time, inp, hr_coord, cell):
        with torch.no_grad():
            srno_feat = self.srno(inp, hr_coord, cell)
        srno_feat = srno_feat.detach()
        x = torch.cat([x, srno_feat], dim=1)

        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)


# states = torch.load('/home/aiseon/Downloads/save2/_train_edsr-sronet/epoch-best.pth', weights_only=True)


# class Model(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.Unet = UNet(config)

#         self.srno = SRNO(config)
#         self.srno.load_state_dict(states['model'])
#         self.srno.eval()

#     def forward(self, x, time, inp, hr_coord, cell):
#         with torch.no_grad():
#             srno_feat = self.srno(inp, hr_coord, cell)
#         srno_feat = srno_feat.detach()
#         x = torch.cat([x, srno_feat], dim=1)

#         return self.Unet(x, time)

        
        
