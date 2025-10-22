import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import modules
logger = logging.getLogger('base')
####################
# initialize
####################
import torch
import torch.nn as nn



logger = logging.getLogger('base')


class DiffusionNet(nn.Module):
    """
    双输入（vis, ir），输出一个融合图像（SR）。
    通过 opt 构建网络，如果初始化时没传 opt，
    就会使用 set_default_opt(opt) 设置的全局默认配置。
    """
    _default_opt = None   # 类变量，全局共享 opt

    @classmethod
    def set_default_opt(cls, opt):
        """
        设置全局默认 opt，在训练入口调用一次即可。
        """
        cls._default_opt = opt

    def __init__(self, opt=None, device="cuda"):
        super(DiffusionNet, self).__init__()
        self.device = device
        # 如果没传 opt，就用全局默认
        if opt is None:
            if DiffusionNet._default_opt is None:
                raise RuntimeError("❌ 请先调用 DiffusionNet.set_default_opt(opt)")
            opt = DiffusionNet._default_opt

        # 调用本文件里的 define_G 构建网络
        self.netG = define_G(opt).to(device)
        self.schedule_phase = None

        # 👇 这里补充初始化噪声调度（关键）
        self.netG.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], device
        )

    def forward(self, vis, ir, continous=False, mode="default"):
        """
        输入: vis, ir (tensor)
        输出: 融合图像 SR
        """
        data = {"vis": vis, "ir": ir}

        if mode == "default":
            SR, _, _ = self.netG.super_resolution(data, continous)
        elif mode == "ddim":
            SR, _, _ = self.netG.super_resolution_ddim(data, continous)
        else:
            raise NotImplementedError(f"Unknown mode: {mode}")
        if SR.dim() == 3:
            SR = SR.unsqueeze(0)

        return SR, None, None, None, None




def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))

# Generator
def define_G(opt):
    model_opt = opt['model']
    if model_opt['which_model_G'] == 'ddpm':
        from .ddpm_modules import diffusion, unet
    elif model_opt['which_model_G'] == 'sr3':
        from .sr3_modules import diffusion, unet
    elif model_opt['which_model_G'] == 'diffif':
        from .diffif_modules import diffusion, unet, finetune_arch

    if ('norm_groups' not in model_opt['unet_denoising']) or model_opt['unet_denoising']['norm_groups'] is None:
        model_opt['unet_denoising']['norm_groups'] = 24

    model = unet.UNet(
        in_channel=model_opt['unet_denoising']['in_channel'],
        out_channel=model_opt['unet_denoising']['out_channel'],
        norm_groups=model_opt['unet_denoising']['norm_groups'],
        inner_channel=model_opt['unet_denoising']['inner_channel'],
        channel_mults=model_opt['unet_denoising']['channel_multiplier'],
        attn_res=model_opt['unet_denoising']['attn_res'],
        res_blocks=model_opt['unet_denoising']['res_blocks'],
        dropout=model_opt['unet_denoising']['dropout']
    )

    model_refinement_fn = finetune_arch.Restormer_fn(
        in_channel=model_opt['unet_refine']['in_channel'],
        out_channel=model_opt['unet_refine']['out_channel']
    )

    netG = diffusion.GaussianDiffusion(
        denoise_fn=model,
        refinement_fn=model_refinement_fn,
        image_size=model_opt['diffusion'].get('image_size', 256),
        channels=model_opt['diffusion']['channels'],
        loss_type='l1',  # opt: l1/l2
        conditional=model_opt['diffusion']['conditional'],
        schedule_opt=model_opt['beta_schedule']['train']
    )

    if opt['phase'] == 'train':
        init_weights(netG, init_type='orthogonal')

    # ❌ 删除 GPU 相关逻辑，不处理 gpu_ids/distributed
    return netG

