import torch
import torch.nn as nn
import torch.nn.functional as F 
from .segformer_head import SegFormerHead
from . import mix_transformer
from timm.models.layers import trunc_normal_
import math
import functools
from core.NewFusionNet import FusionNet
from core.Diff import DiffusionNet
# from core.NewFusionNet import FusionNet
from core.NewFusionNet import Restormer_Encoder, Restormer_Decoder
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from net import CDDFuse as CDDFusion_Network
from core.gesenet import GeSeNet
# 你已有的依赖（本文件里应已存在）
# from core.somewhere import WeTr, PixelDiscriminator, GANLoss, CDDFusion_Network ...
# 这里我们只引入你的新融合网络
from core.NewFusionNet import Restormer_Encoder, Restormer_Decoder





class WeTr(nn.Module):
    def __init__(self, backbone, num_classes=2, embedding_dim=256, pretrained=None):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.backbone = backbone
        self.feature_strides = [4, 8, 16, 32]
        #self.in_channels = [32, 64, 160, 256]
        #self.in_channels = [64, 128, 320, 512]

        self.encoder = getattr(mix_transformer, backbone)(in_channels=3)
        self.in_channels = self.encoder.embed_dims
        ## initilize encoder
        if pretrained:
            state_dict = torch.load('pretrained/'+backbone+'.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict,)

        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels, embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        
        self.classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.num_classes, kernel_size=1, bias=False)

    def initialize(self,):
        state_dict = torch.load('pretrained/' + self.backbone + '.pth')
        state_dict.pop('head.weight')
        state_dict.pop('head.bias')
        self.encoder.load_state_dict(state_dict, )
    def _forward_cam(self, x):
        
        cam = F.conv2d(x, self.classifier.weight)
        cam = F.relu(cam)
        
        return cam

    def get_param_groups(self):

        param_groups = [[], [], []] # 
        
        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        for param in list(self.decoder.parameters()):

            param_groups[2].append(param)
        
        param_groups[2].append(self.classifier.weight)

        return param_groups

    def forward(self, x):

        _x = self.encoder(x)
        _x1, _x2, _x3, _x4 = _x
        cls = self.classifier(_x4)

        return self.decoder(_x)
def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


import torch
import torch.nn as nn
import torch.nn.functional as F
class DRDB(nn.Module):
    def __init__(self, in_ch=64, growth_rate=32):
        super(DRDB, self).__init__()
        in_ch_ = in_ch
        self.Dcov1 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov2 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov3 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov4 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov5 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        print(in_ch_,in_ch)
        self.conv = nn.Conv2d(in_ch_, in_ch, 1, padding=0)

    def forward(self, x):
        x1 = self.Dcov1(x)
        x1 = F.relu(x1)
        x1 = torch.cat([x, x1], dim=1)

        x2 = self.Dcov2(x1)
        x2 = F.relu(x2)
        x2 = torch.cat([x1, x2], dim=1)

        x3 = self.Dcov3(x2)
        x3 = F.relu(x3)
        x3 = torch.cat([x2, x3], dim=1)

        x4 = self.Dcov4(x3)
        x4 = F.relu(x4)
        x4 = torch.cat([x3, x4], dim=1)

        x5 = self.Dcov5(x4)
        x5 = F.relu(x5)
        x5 = torch.cat([x4, x5], dim=1)

        x6 = self.conv(x5)
        out = x + F.relu(x6)
        return out

class Fusion_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, 3, padding=1)
        self.DRDB1 = DRDB(in_ch=64)
        self.DRDB2 = DRDB(in_ch=64)
        # self.DRDB3 = DRDB(in_ch=64)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv21 = nn.Conv2d(32, 1, 3, padding=1)
        self.relu = nn.PReLU()
    def forward(self, ir, vis):
        ir = ir[:, 0:1, :, :]  # 确保是单通道输入
        vis = vis[:, 0:1, :, :]  # 同上，单通道输入
        x1 = self.conv1(torch.cat([ir,vis],dim=1))
        x1 = self.relu(x1)
        f1 = self.DRDB1(x1)
        f2 = self.DRDB2(f1)
        # f2 = self.DRDB3(f2)
        f_final = self.relu(self.conv2(f2))
        f_final = self.relu(self.conv21(f_final))
        ones = torch.ones_like(f_final)
        zeros = torch.zeros_like(f_final)
        f_final = torch.where(f_final > ones, ones, f_final)
        f_final = torch.where(f_final < zeros, zeros, f_final)
        # new encode
        f_final = (f_final - torch.min(f_final)) / (
                torch.max(f_final) - torch.min(f_final)
        )
        return f_final

class SKFF(nn.Module):
  def __init__(self, in_channels, height=3, reduction=8, bias=False):
    super(SKFF, self).__init__()

    self.height = height
    d = max(int(in_channels / reduction), 4)

    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

    self.fcs = nn.ModuleList([])
    for i in range(self.height):
      self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))
    self.softmax = nn.Softmax(dim=1)

  def forward(self, inp_feats):
    batch_size = inp_feats[0].shape[0]
    n_feats = inp_feats[0].shape[1]

    inp_feats = torch.cat(inp_feats, dim=1)
    inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

    feats_U = torch.sum(inp_feats, dim=1)
    feats_S = self.avg_pool(feats_U)
    feats_Z = self.conv_du(feats_S)

    attention_vectors = [fc(feats_Z) for fc in self.fcs]
    attention_vectors = torch.cat(attention_vectors, dim=1)
    attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
    # stx()
    attention_vectors = self.softmax(attention_vectors)

    feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

    return feats_V
import numpy as np
class Fusion_Network2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, 3, padding=1)
        self.DRDB1 = DRDB()
        self.DRDB2 = DRDB()
        self.conv2 = nn.Conv2d(64, 1, 3, padding=1)
        self.relu = nn.PReLU()
        self.skff = SKFF(64,2)
        self.skff2 = SKFF(64, 2)
        self.conv3 = nn.Conv2d(64, 64, 1, padding=0)
        self.conv4 = nn.Conv2d(128, 64, 1, padding=0)

    def forward(self, ir, vis, out1, out2):
        # print(np.shape(out1),'----------------')
        vis = vis[:,0:1,:,:]
        ir = ir[:, 0:1, :, :]
        x1 = self.conv1(torch.cat([ir,vis],dim=1))
        x1 = self.relu(x1)
        f1 = self.DRDB1(x1)
        f1 = self.skff([f1,self.conv3(out1)])
        f2 = self.DRDB2(f1)
        f2 = self.skff2([f2,self.conv4(out2)])

        f_final = self.relu(self.conv2(f2))
        # ones = torch.ones_like(f_final)
        # zeros = torch.zeros_like(f_final)
        # f_final = torch.where(f_final > ones, ones, f_final)
        # f_final = torch.where(f_final < zeros, zeros, f_final)
        # # new encode
        f_final = (f_final - torch.min(f_final)) / (
                torch.max(f_final) - torch.min(f_final)
        )
        return f_final


from.loss import IQALoss

### 1215 new architecture remove architecture retrain the network + discriminator
### Gradient_Norm or other multi-task learning schemes
###

class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


# class Network_IJCAI(nn.Module):

#     def __init__(self, f_loss, segloss,backbone, num_classes=2, embedding_dim=256, pretrained=None):
#         super(Network_IJCAI, self).__init__()

#         self.fusion_nums = 2
#         self.seg_nums = 2
#         self.fusion_channel = 48
#         self.seg_channel = 64
#         self._criterion = f_loss
#         # self._criterion_lower = IQALoss()
#         self._criterion_gan = GANLoss('wgangp')
#         self.seg_loss = segloss

#         self.enhance_net = CDDFusion_Network()
#         self.denoise_net = WeTr(backbone,num_classes,embedding_dim,pretrained)

#         self.discriminator = PixelDiscriminator(1)
#         self.mean = [0.485, 0.485, 0.485]  # 灰度图的平均值
#         self.std = [0.229, 0.229, 0.229]
#     def forward(self, ir, vis):
#         # vis = RGB2YCrCb(vis)
#         ir = ir[:, :1, :, :]  # 确保 ir 是单通道
#         vis = vis[:, :1, :, :]  # 确保 vis 是单通道
#         fused = self.enhance_net.forward(ir, vis)
#         fused_3channel = fused.repeat(1, 3, 1, 1)  # 从单通道扩展为三通道
#         # # print(f"fused shape: {fused.shape}") 
#         # # print(f"fused_3channel shape: {fused_3channel.shape}")  # 输出 [batch_size, 3, 256, 256]
#         # # print(f"vis shape: {vis.shape}") 
#         # try:
#         #     fused_seg = torch.cat((fused_3channel, vis[:, 1:2, :, :], vis[:, 2:, :, :]), dim=1)
#         # except IndexError:
#         # # 如果 `vis` 是单通道灰度图，仅使用 `fused_3channel`
#         #     fused_seg = fused
#         #print(f"{fused_seg.shape}")
#         #fused_seg = YCrCb2RGB(fused_seg)
#         #print(f"fused_seg:{fused_seg.shape}")
#         fused_seg = fused_3channel
#         ones = torch.ones_like(fused_seg)
#         zeros = torch.zeros_like(fused_seg)
#         fused_seg = torch.where(fused_seg > ones, ones, fused_seg)
#         fused_seg = torch.where(fused_seg < zeros, zeros, fused_seg)
        
#         # new encode
#         fused_seg = (fused_seg - torch.min(fused_seg)) / (
#                 torch.max(fused_seg) - torch.min(fused_seg)
#         )
#         fused_seg1 = fused_seg
#         #print(f"fused_seg1:{fused_seg1.shape}")
#         torch_norma = fused_seg1*255 
#         #print(f"torch_norma:{torch_norma.shape}")
#         num_channels = torch_norma.shape[1]
        
#         for index in range(num_channels):  # 循环次数取决于图像的通道数
#             torch_norma[:, index, :, :] = (torch_norma[:, index, :, :] - self.mean[index]) / self.std[index]
        
#         seg_map = self.denoise_net(torch_norma)
#         return fused, seg_map

#     def forward_fusion(self, ir, vis):
#         #vis = RGB2YCrCb(vis)
#         ir = ir[:, :1, :, :]  # 确保 ir 是单通道
#         vis = vis[:, :1, :, :]  # 确保 vis 是单通道
#         fused = self.enhance_net.forward(ir, vis)
#         return fused
#     def forward_object(self, ir, vis):
#         #vis = RGB2YCrCb(vis)
#         ir = ir[:, :1, :, :]  # 确保 ir 是单通道
#         vis = vis[:, :1, :, :]  # 确保 vis 是单通道
#         fused = self.enhance_net.forward(ir, vis)
#         ones = torch.ones_like(fused)
#         zeros = torch.zeros_like(fused)
#         fused = torch.where(fused > ones, ones, fused)
#         fused = torch.where(fused < zeros, zeros, fused)
#         # new encode
#         fused = (fused - torch.min(fused)) / (
#                 torch.max(fused) - torch.min(fused)
#         )
#         fused_seg = torch.cat((fused,fused,fused ), dim=1)
#         # fused_seg = YCrCb2RGB(fused_seg)
#         ones = torch.ones_like(fused_seg)
#         zeros = torch.zeros_like(fused_seg)
#         fused_seg = torch.where(fused_seg > ones, ones, fused_seg)
#         fused_seg = torch.where(fused_seg < zeros, zeros, fused_seg)
#         # new encode
#         fused_seg = (fused_seg - torch.min(fused_seg)) / (
#                 torch.max(fused_seg) - torch.min(fused_seg)
#         )
#         fused_seg1 = fused_seg
#         torch_norma = fused_seg1 * 255
#         for index in range(3):
#             torch_norma[:, index, :, :] = (torch_norma[:, index, :, :] - self.mean[index]) / self.std[index]

#         seg_map = self.denoise_net(torch_norma)
#         return fused,seg_map
#     def _loss(self, ir, vis,mask,labels):
#         fused_img, seg_map = self(ir,vis)
#         #vis = RGB2YCrCb(vis)
#         outputs = F.interpolate(seg_map, size=labels.shape[1:], mode='bilinear', align_corners=False)
#         enhance_loss = self._criterion(ir, vis, fused_img,mask)
#         denoise_loss = self.seg_loss(outputs,labels.type(torch.long))
#         # vis = RGB2YCrCb(vis)
#         ### update the discriminator of fused network
#         pred_fake = self.discriminator(fused_img)
#         loss_D_fake = self._criterion_gan(pred_fake, False)
#         # Real
#         # real_AB = torch.cat((real_A, real_B), 1)
#         pred_real = self.discriminator(mask)
#         loss_D_real = self._criterion_gan(pred_real, True)
#         # combine loss and calculate gradients
#         loss_D = (loss_D_fake + loss_D_real) * 0.5

#         return enhance_loss*0.1+denoise_loss*4+loss_D*0.01

#     # def _loss(self, ir, vis,mask,labels):
#     #     # fused_img, seg_map = self(ir,vis)
#     #     # vis = RGB2YCrCb(vis)
#     #     # outputs = F.interpolate(seg_map, size=labels.shape[1:], mode='bilinear', align_corners=False)
#     #     # enhance_loss = self._criterion(ir, vis, fused_img,mask)
#     #     # denoise_loss = self.seg_loss(outputs,labels.type(torch.long))
#     #     # # vis = RGB2YCrCb(vis)
#     #     # ### update the discriminator of fused network
#     #     # pred_fake = self.discriminator(fused_img)
#     #     # loss_D_fake = self._criterion_gan(pred_fake, False)
#     #     # # Real
#     #     # # real_AB = torch.cat((real_A, real_B), 1)
#     #     # pred_real = self.discriminator(mask)
#     #     # loss_D_real = self._criterion_gan(pred_real, True)
#     #     # # combine loss and calculate gradients
#     #     # loss_D = (loss_D_fake + loss_D_real) * 0.5
#     #     #
#     #     # return enhance_loss*0.1+denoise_loss*4+loss_D*0.01

#     def _fusion_loss_lower(self, ir, vis,mask):
#         fused_img, seg_map = self(ir,vis)
#         #vis = RGB2YCrCb(vis)
#         enhance_loss = self._criterion(ir, vis, fused_img,mask)
#         # vis = RGB2YCrCb(vis)
#         ### update the discriminator of fused network
#         pred_fake = self.discriminator(fused_img)
#         loss_D_fake = self._criterion_gan(pred_fake, False)
#         # Real
#         # real_AB = torch.cat((real_A, real_B), 1)
#         pred_real = self.discriminator(mask)
#         loss_D_real = self._criterion_gan(pred_real, True)
#         # combine loss and calculate gradients
#         loss_D = (loss_D_fake + loss_D_real) * 0.5

#         return enhance_loss +loss_D*0.5
#     ############ Update_upper_level_features
#     def _fusion_loss(self, ir, vis, mask,):
#         fused_img = self.forward_fusion(ir, vis)
#         #vis = RGB2YCrCb(vis)
#         enhance_loss = self._criterion(ir, vis, fused_img,mask)
#         pred_fake = self.discriminator(fused_img)
#         loss_D_fake = self._criterion_gan(pred_fake, True)
#         loss_D = loss_D_fake
#         return enhance_loss + loss_D*0.5
#     def _fusion_loss_wogan(self, ir, vis, mask,):
#         fused_img = self.forward_fusion(ir, vis)
#         #vis = RGB2YCrCb(vis)
#         enhance_loss = self._criterion(ir, vis, fused_img,mask)
#         # pred_fake = self.discriminator(fused_img)
#         # loss_D_fake = self._criterion_gan(pred_fake, True)
#         # loss_D = loss_D_fake
#         return enhance_loss
#     def _detection_loss(self, ir, vis, labels):
#         fused_img, seg_map = self.forward_object(ir, vis)
#         outputs = F.interpolate(seg_map, size=labels.shape[1:], mode='bilinear', align_corners=False)
#         denoise_loss = self.seg_loss(outputs, labels.type(torch.long))

#         return denoise_loss

#     def enhance_net_parameters(self):
#         return self.enhance_net.parameters()

#     def denoise_net_parameters(self):
#         return self.denoise_net.parameters()


class Network_IJCAI(nn.Module):
    def __init__(self, f_loss, segloss, backbone, num_classes=2, embedding_dim=256, pretrained=None):
    
        super(Network_IJCAI, self).__init__()

        self.fusion_nums = 2
        self.seg_nums = 2
        self.fusion_channel = 48
        self.seg_channel = 64
        self._criterion = f_loss
        self._criterion_gan = GANLoss('wgangp')
        self.seg_loss = segloss

        # ✅ 替换增强网络为 FusionNet
        self.enhance_net = FusionNet()
        # self.enhance_net = CDDFusion_Network()
        # self.enhance_net = GeSeNet(output=1)
        



        self.denoise_net = WeTr(backbone, num_classes, embedding_dim, pretrained)
        self.discriminator = PixelDiscriminator(1)
        self.mean = [0.485, 0.485, 0.485]
        self.std = [0.229, 0.229, 0.229]

    def forward(self, ir, vis):
        ir = ir[:, :1, :, :]
        vis = vis[:, :1, :, :]

        # ✅ 只取 fused_img，其余值后续可用
        fused, _, _, _, _ = self.enhance_net(ir, vis)

        fused_3channel = fused.repeat(1, 3, 1, 1)

        ones = torch.ones_like(fused_3channel)
        zeros = torch.zeros_like(fused_3channel)
        fused_seg = torch.where(fused_3channel > ones, ones, fused_3channel)
        fused_seg = torch.where(fused_seg < zeros, zeros, fused_seg)

        fused_seg = (fused_seg - torch.min(fused_seg)) / (
            torch.max(fused_seg) - torch.min(fused_seg)
        )

        torch_norma = fused_seg * 255
        num_channels = torch_norma.shape[1]
        for index in range(num_channels):
            torch_norma[:, index, :, :] = (torch_norma[:, index, :, :] - self.mean[index]) / self.std[index]

        seg_map = self.denoise_net(torch_norma)
        return fused, seg_map

    def forward_fusion(self, ir, vis):
        ir = ir[:, :1, :, :]
        vis = vis[:, :1, :, :]
        fused, _, _, _, _ = self.enhance_net(ir, vis)
        return fused

    def forward_object(self, ir, vis):
        ir = ir[:, :1, :, :]
        vis = vis[:, :1, :, :]
        fused, _, _, _, _ = self.enhance_net(ir, vis)

        ones = torch.ones_like(fused)
        zeros = torch.zeros_like(fused)
        fused = torch.where(fused > ones, ones, fused)
        fused = torch.where(fused < zeros, zeros, fused)

        fused = (fused - torch.min(fused)) / (
            torch.max(fused) - torch.min(fused)
        )

        fused_seg = torch.cat((fused, fused, fused), dim=1)

        ones = torch.ones_like(fused_seg)
        zeros = torch.zeros_like(fused_seg)
        fused_seg = torch.where(fused_seg > ones, ones, fused_seg)
        fused_seg = torch.where(fused_seg < zeros, zeros, fused_seg)

        fused_seg = (fused_seg - torch.min(fused_seg)) / (
            torch.max(fused_seg) - torch.min(fused_seg)
        )

        torch_norma = fused_seg * 255
        for index in range(3):
            torch_norma[:, index, :, :] = (torch_norma[:, index, :, :] - self.mean[index]) / self.std[index]

        seg_map = self.denoise_net(torch_norma)
        return fused, seg_map

    def _loss(self, ir, vis, mask, labels):
        fused_img, seg_map = self(ir, vis)
        outputs = F.interpolate(seg_map, size=labels.shape[1:], mode='bilinear', align_corners=False)

        enhance_loss = self._criterion(ir, vis, fused_img, mask)
        denoise_loss = self.seg_loss(outputs, labels.type(torch.long))

        pred_fake = self.discriminator(fused_img)
        loss_D_fake = self._criterion_gan(pred_fake, False)

        pred_real = self.discriminator(mask)
        loss_D_real = self._criterion_gan(pred_real, True)

        loss_D = (loss_D_fake + loss_D_real) * 0.5
        # return enhance_loss * 0.1 + denoise_loss * 4 + loss_D * 0.01
        return enhance_loss * 1 + denoise_loss * 1 + loss_D * 0.01

    def _fusion_loss_lower(self, ir, vis, mask):
        fused_img, seg_map = self(ir, vis)
        enhance_loss = self._criterion(ir, vis, fused_img, mask)

        pred_fake = self.discriminator(fused_img)
        loss_D_fake = self._criterion_gan(pred_fake, False)

        pred_real = self.discriminator(mask)
        loss_D_real = self._criterion_gan(pred_real, True)

        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return enhance_loss + loss_D * 0.5

    def _fusion_loss(self, ir, vis, mask):
        fused_img = self.forward_fusion(ir, vis)
        enhance_loss = self._criterion(ir, vis, fused_img, mask)

        pred_fake = self.discriminator(fused_img)
        loss_D_fake = self._criterion_gan(pred_fake, True)
        loss_D = loss_D_fake

        return enhance_loss + loss_D * 0.5

    def _fusion_loss_wogan(self, ir, vis, mask):
        fused_img = self.forward_fusion(ir, vis)
        enhance_loss = self._criterion(ir, vis, fused_img, mask)
        return enhance_loss

    def _detection_loss(self, ir, vis, labels):
        fused_img, seg_map = self.forward_object(ir, vis)
        outputs = F.interpolate(seg_map, size=labels.shape[1:], mode='bilinear', align_corners=False)
        denoise_loss = self.seg_loss(outputs, labels.type(torch.long))
        return denoise_loss

    def enhance_net_parameters(self):
        return self.enhance_net.parameters()

    def denoise_net_parameters(self):
        return self.denoise_net.parameters()



class Mean(nn.Module):
    def __init__(self, ):
        super(Mean, self).__init__()

        self.fusion_nums = 2
        self.seg_nums = 2
        self.fusion_channel = 48
        self.seg_channel = 64
        # self._criterion_lower = IQALoss()
        self.mean =[123.675, 116.28, 103.53]
        self.std = [58.395, 57.12, 57.375]
    def forward(self, mask, vis):
        vis = RGB2YCrCb(vis)
        # fused = self.enhance_net.forward(ir, vis)
        # ir_mask, vis_mask = self.discriminator(fused)
        ##Fixed some c
        # fused = ir[:, 0:1, :, :] + vis[:, 0:1, :, :]
        fused_seg = torch.cat((mask, vis[:, 1:2, :, :], vis[:, 2:, :, :]), dim=1)
        fused_seg = YCrCb2RGB(fused_seg)
        ones = torch.ones_like(fused_seg)
        zeros = torch.zeros_like(fused_seg)
        fused_seg = torch.where(fused_seg > ones, ones, fused_seg)
        fused_seg = torch.where(fused_seg < zeros, zeros, fused_seg)
        # new encode
        fused_seg = (fused_seg - torch.min(fused_seg)) / (
                torch.max(fused_seg) - torch.min(fused_seg)
        )
        # x_ir,x_vis = self.discriminator(fused)
        ### add augmentation

        return fused_seg
class Network2(nn.Module):

    def __init__(self, f_loss, segloss,backbone, num_classes=20, embedding_dim=256, pretrained=None):
        super(Network2, self).__init__()

        self.fusion_nums = 2
        self.seg_nums = 2
        self.fusion_channel = 48
        self.seg_channel = 64
        self._criterion = f_loss
        # self._criterion_lower = IQALoss()
        self.seg_loss = segloss
        self.enhance_net = Fusion_Network()
        self.denoise_net = WeTr(backbone,num_classes,embedding_dim,pretrained)
        self.discriminator = Discriminator()
        self.mean =[123.675, 116.28, 103.53]
        self.std = [58.395, 57.12, 57.375]
    def forward(self, ir, vis):
        vis = RGB2YCrCb(vis)
        # fused = self.enhance_net.forward(ir, vis)
        # ir_mask, vis_mask = self.discriminator(fused)
        ##Fixed some c
        fused = ir[:, 0:1, :, :] + vis[:, 0:1, :, :]/2
        fused_seg = torch.cat((fused, vis[:, 1:2, :, :], vis[:, 2:, :, :]), dim=1)
        fused_seg = YCrCb2RGB(fused_seg)
        ones = torch.ones_like(fused_seg)
        zeros = torch.zeros_like(fused_seg)
        fused_seg = torch.where(fused_seg > ones, ones, fused_seg)
        fused_seg = torch.where(fused_seg < zeros, zeros, fused_seg)
        # new encode
        fused_seg = (fused_seg - torch.min(fused_seg)) / (
                torch.max(fused_seg) - torch.min(fused_seg)
        )
        # x_ir,x_vis = self.discriminator(fused)
        ### add augmentation
        fused_seg1 = fused_seg
        torch_norma = fused_seg1*255
        for index in range(3):
            torch_norma[:,index,:,:] = (torch_norma[:,index,:,:] - self.mean[index])/self.std[index]

        seg_map = self.denoise_net(torch_norma)
        return fused,fused_seg,seg_map


    def _loss(self, ir, vis,mask,labels):
        fused_img, seg1, seg2 = self(ir, vis)
        outputs = F.interpolate(seg2, size=labels.shape[1:], mode='bilinear', align_corners=False)
        denoise_loss = self.seg_loss(outputs,labels.type(torch.long))
        # vis = RGB2YCrCb(vis)
        return denoise_loss*1.25




    def _detection_loss(self, ir, vis, labels):
        fused_img, seg_map = self(ir, vis)

        outputs = F.interpolate(seg_map, size=labels.shape[1:], mode='bilinear', align_corners=False)
        # seg_loss = criterion(outputs, labels.type(torch.long))
        denoise_loss = self.seg_loss(outputs, labels.type(torch.long))        # vis = RGB2YCrCb(vis)
        return denoise_loss

    def enhance_net_parameters(self):
        return self.enhance_net.parameters()

    def denoise_net_parameters(self):
        return self.denoise_net.parameters()




class Network_fused(nn.Module):

    def __init__(self, segloss,backbone, num_classes=20, embedding_dim=256, pretrained=None):
        super(Network_fused, self).__init__()

        self.fusion_nums = 2
        self.seg_nums = 2
        self.fusion_channel = 48
        self.seg_channel = 64
        self.seg_loss = segloss
        self.denoise_net = WeTr(backbone,num_classes,embedding_dim,pretrained)
        self.mean =[123.675, 116.28, 103.53]
        self.std = [58.395, 57.12, 57.375]
    def forward(self, fused):
        seg_map = self.denoise_net(fused)
        return seg_map
    def _loss(self, fused,labels):
        seg2 = self(fused)
        outputs = F.interpolate(seg2, size=labels.shape[1:], mode='bilinear', align_corners=False)
        denoise_loss = self.seg_loss(outputs,labels.type(torch.long))
        return denoise_loss


    def enhance_net_parameters(self):
        return self.enhance_net.parameters()

    def denoise_net_parameters(self):
        return self.denoise_net.parameters()

## Introducing two corss attention modules and concate the feature for the final fusion


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        # self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv3 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2, segfeature):
        B, N, C = x1.shape
        q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        # q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]

        # k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        # k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k3, v3 = self.kv3(segfeature).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        # ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        # ctx1 = ctx1.softmax(dim=-2)
        # ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        # ctx2 = ctx2.softmax(dim=-2)

        ctx3 = (k3.transpose(-2, -1) @ v3) * self.scale
        ctx3 = ctx3.softmax(dim=-2)


        x1 = (q1 @ ctx3).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        x2 = (q2 @ ctx3).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()

        return x1, x2

class CrossAttention2(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttention2, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        # self.kv3 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2, segfeature):
        B, N, C = x1.shape
        # q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        # q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q3 = segfeature.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]

        k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        # k3, v3 = self.kv3(segfeature).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)
        #
        # ctx3 = (k3.transpose(-2, -1) @ v3) * self.scale
        # ctx3 = ctx3.softmax(dim=-2)


        x1 = (q3 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        x2 = (q3 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()

        return x1, x2
class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=8, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj3 = nn.Linear(dim, dim // reduction * 2)

        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.act3 = nn.ReLU(inplace=True)

        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.cross_attn2 = CrossAttention2(dim // reduction, num_heads=num_heads)

        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        # self.end_proj2 = nn.Linear(dim // reduction * 2, dim)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2,segfeature):
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        y3, u3 = self.act3(self.channel_proj3(segfeature)).chunk(2, dim=-1)

        v1, v2 = self.cross_attn(u1, u2,u3)
        z1, z2 = self.cross_attn2(y1,y2,y3)
        y1 = torch.cat((z1, v1), dim=-1)
        y2 = torch.cat((z2, v2), dim=-1)
        out_x1 = self.norm1(x1 + self.end_proj1(y1))
        out_x2 = self.norm2(x2 + self.end_proj2(y2))
        return out_x1, out_x2

class CrossPath_M(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=8, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj3 = nn.Linear(dim, dim // reduction * 2)

        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.act3 = nn.ReLU(inplace=True)

        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        # self.cross_attn2 = CrossAttention2(dim // reduction, num_heads=num_heads)

        self.end_proj1 = nn.Linear(dim, dim)
        self.end_proj2 = nn.Linear(dim, dim)
        # self.end_proj2 = nn.Linear(dim // reduction * 2, dim)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2,segfeature):
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        y3, u3 = self.act3(self.channel_proj3(segfeature)).chunk(2, dim=-1)

        v1, v2 = self.cross_attn(u1, u2,u3)
        # z1, z2 = self.cross_attn2(y1,y2,y3)
        # y1 = torch.cat((z1, v1), dim=-1)
        # y2 = torch.cat((z2, v2), dim=-1)
        out_x1 = self.norm1(x1 + self.end_proj1(v1))
        out_x2 = self.norm2(x2 + self.end_proj2(v2))
        return out_x1, out_x2

class CrossPath_S(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=8, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj3 = nn.Linear(dim, dim // reduction * 2)

        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.act3 = nn.ReLU(inplace=True)

        # self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.cross_attn2 = CrossAttention2(dim // reduction, num_heads=num_heads)

        self.end_proj1 = nn.Linear(dim, dim)
        self.end_proj2 = nn.Linear(dim, dim)
        # self.end_proj2 = nn.Linear(dim // reduction * 2, dim)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2, segfeature):
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        y3, u3 = self.act3(self.channel_proj3(segfeature)).chunk(2, dim=-1)

        # v1, v2 = self.cross_attn(u1, u2, u3)
        z1, z2 = self.cross_attn2(y1,y2,y3)
        # y1 = torch.cat((z1, v1), dim=-1)
        # y2 = torch.cat((z2, v2), dim=-1)
        out_x1 = self.norm1(x1 + self.end_proj1(z1))
        out_x2 = self.norm2(x2 + self.end_proj2(z2))
        return out_x1, out_x2
class FeatureFusionModule(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=8, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        # self.channel_emb = ChannelEmbed(in_channels=dim * 2, out_channels=dim, reduction=reduction,
        #                                 norm_layer=norm_layer)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2,segfeature):
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x3 = segfeature.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2,x3)
        # merge = torch.cat((x1, x2), dim=-1)
        # merge = self.channel_emb(merge, H, W)
        x1 = x1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x2 = x2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x1,x2



class FeatureFusionModule_SoAM(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=8, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath_S(dim=dim, reduction=reduction, num_heads=num_heads)
        # self.channel_emb = ChannelEmbed(in_channels=dim * 2, out_channels=dim, reduction=reduction,
        #                                 norm_layer=norm_layer)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2,segfeature):
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x3 = segfeature.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2,x3)
        # merge = torch.cat((x1, x2), dim=-1)
        # merge = self.channel_emb(merge, H, W)
        x1 = x1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x2 = x2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x1,x2


class FeatureFusionModule_MoAM(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=8, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath_M(dim=dim, reduction=reduction, num_heads=num_heads)
        # self.channel_emb = ChannelEmbed(in_channels=dim * 2, out_channels=dim, reduction=reduction,
        #                                 norm_layer=norm_layer)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2,segfeature):
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x3 = segfeature.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2,x3)
        # merge = torch.cat((x1, x2), dim=-1)
        # merge = self.channel_emb(merge, H, W)
        x1 = x1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x2 = x2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x1,x2

class CrossPath_showAttention(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=8, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj3 = nn.Linear(dim, dim // reduction * 2)

        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.act3 = nn.ReLU(inplace=True)

        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.cross_attn2 = CrossAttention2(dim // reduction, num_heads=num_heads)

        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        # self.end_proj2 = nn.Linear(dim // reduction * 2, dim)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2,segfeature):
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        y3, u3 = self.act3(self.channel_proj3(segfeature)).chunk(2, dim=-1)

        v1, v2 = self.cross_attn(u1, u2,u3)
        z1, z2 = self.cross_attn2(y1,y2,y3)
        y1 = torch.cat((z1, v1), dim=-1)
        y2 = torch.cat((z2, v2), dim=-1)
        out_x1 = self.norm1(x1 + self.end_proj1(y1))
        out_x2 = self.norm2(x2 + self.end_proj2(y2))


        return out_x1, out_x2, [v1,z1,z2,v2]
class FeatureFusionModule_ShowAttention(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=8, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath_showAttention(dim=dim, reduction=reduction, num_heads=num_heads)
        # self.channel_emb = ChannelEmbed(in_channels=dim * 2, out_channels=dim, reduction=reduction,
        #                                 norm_layer=norm_layer)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2,segfeature):
        B, C, H, W = x1.shape
        x1_old = x1.flatten(2).transpose(1, 2)
        x2_old = x2.flatten(2).transpose(1, 2)
        x3_old = segfeature.flatten(2).transpose(1, 2)
        x1, x2, list_attention = self.cross(x1_old, x2_old,x3_old)
        x1 = x1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x2 = x2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        z1 = list_attention[0].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        v1 = list_attention[1].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        z2 = list_attention[2].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        v2 = list_attention[3].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x1_old = x1_old.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x2_old = x2_old.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x3_old = x3_old.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x1_old_max = torch.tensor(x1_old)
        # print('----------',np.shape(x1_old_max))
        x2_old_max = torch.tensor(x2_old)
        x3_old_max = torch.tensor(x3_old)
        x1_max = torch.tensor(x1)
        x2_max = torch.tensor(x2)

        v1_max = torch.tensor(v1)
        z1_max = torch.tensor(z1)
        z2_max = torch.tensor(z2)
        v2_max = torch.tensor(v2)

        return x1,x2,[x1_old_max,x2_old_max,x3_old_max,x1_max,x2_max,v1_max,v2_max,z1_max,z2_max]

class Fusion_Network3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_ir = nn.Conv2d(1, 32, 3, padding=1)
        self.conv1_vis = nn.Conv2d(1, 32, 3, padding=1)
        self.DRDB1 = DRDB(in_ch=32)
        self.DRDB2 = DRDB(in_ch=32)
        self.DRDB3 = DRDB(in_ch=32)
        self.DRDB4 = DRDB(in_ch=32)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv21 = nn.Conv2d(32, 1, 3, padding=1)
        self.relu = nn.PReLU()
        self.ffm = FeatureFusionModule(32)
        self.ffm2 = FeatureFusionModule(32)
        self.conv3 = nn.Conv2d(64, 32, 1, padding=0)
        self.conv4 = nn.Conv2d(128, 32, 1, padding=0)

    def forward(self, ir, vis, out1, out2):
        # print(np.shape(out1),'----------------')

        ir = ir[:, 0:1, :, :]
        x1 = self.conv1_ir(ir)
        x1 = self.relu(x1)
        x1 =self.DRDB1(x1)
        vis = vis[:, 0:1, :, :]
        x2 = self.conv1_vis(vis)
        x2 = self.relu(x2)
        x2 = self.DRDB2(x2)
        x1,x2 = self.ffm(x1,x2,self.conv3(out1))
        x1 = self.DRDB3(x1)
        x2 = self.DRDB4(x2)
        x1, x2 = self.ffm(x1, x2, self.conv4(out2))
        f_final = self.relu(self.conv2((torch.cat([x1,x2],dim=1))))
        f_final= self.relu(self.conv21(f_final))
        return f_final


class Fusion_Network3_S(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_ir = nn.Conv2d(1, 32, 3, padding=1)
        self.conv1_vis = nn.Conv2d(1, 32, 3, padding=1)
        self.DRDB1 = DRDB(in_ch=32)
        self.DRDB2 = DRDB(in_ch=32)
        self.DRDB3 = DRDB(in_ch=32)
        self.DRDB4 = DRDB(in_ch=32)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv21 = nn.Conv2d(32, 1, 3, padding=1)
        self.relu = nn.PReLU()
        self.ffm = FeatureFusionModule_SoAM(32)
        self.conv3 = nn.Conv2d(64, 32, 1, padding=0)
        self.conv4 = nn.Conv2d(128, 32, 1, padding=0)

    def forward(self, ir, vis, out1, out2):
        # print(np.shape(out1),'----------------')

        ir = ir[:, 0:1, :, :]
        x1 = self.conv1_ir(ir)
        x1 = self.relu(x1)
        x1 =self.DRDB1(x1)
        vis = vis[:, 0:1, :, :]
        x2 = self.conv1_vis(vis)
        x2 = self.relu(x2)
        x2 = self.DRDB2(x2)
        x1,x2 = self.ffm(x1,x2,self.conv3(out1))
        x1 = self.DRDB3(x1)
        x2 = self.DRDB4(x2)
        x1, x2 = self.ffm(x1, x2, self.conv4(out2))
        f_final = self.relu(self.conv2((torch.cat([x1,x2],dim=1))))
        f_final= self.relu(self.conv21(f_final))
        return f_final

class Fusion_Network3_M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_ir = nn.Conv2d(1, 32, 3, padding=1)
        self.conv1_vis = nn.Conv2d(1, 32, 3, padding=1)
        self.DRDB1 = DRDB(in_ch=32)
        self.DRDB2 = DRDB(in_ch=32)
        self.DRDB3 = DRDB(in_ch=32)
        self.DRDB4 = DRDB(in_ch=32)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv21 = nn.Conv2d(32, 1, 3, padding=1)
        self.relu = nn.PReLU()
        self.ffm = FeatureFusionModule_MoAM(32)
        self.conv3 = nn.Conv2d(64, 32, 1, padding=0)
        self.conv4 = nn.Conv2d(128, 32, 1, padding=0)

    def forward(self, ir, vis, out1, out2):
        # print(np.shape(out1),'----------------')

        ir = ir[:, 0:1, :, :]
        x1 = self.conv1_ir(ir)
        x1 = self.relu(x1)
        x1 =self.DRDB1(x1)
        vis = vis[:, 0:1, :, :]
        x2 = self.conv1_vis(vis)
        x2 = self.relu(x2)
        x2 = self.DRDB2(x2)
        x1,x2 = self.ffm(x1,x2,self.conv3(out1))
        x1 = self.DRDB3(x1)
        x2 = self.DRDB4(x2)
        x1, x2 = self.ffm(x1, x2, self.conv4(out2))
        f_final = self.relu(self.conv2((torch.cat([x1,x2],dim=1))))
        f_final= self.relu(self.conv21(f_final))
        return f_final

class Fusion_Network3_obtainattention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_ir = nn.Conv2d(1, 32, 3, padding=1)
        self.conv1_vis = nn.Conv2d(1, 32, 3, padding=1)
        self.DRDB1 = DRDB(in_ch=32)
        self.DRDB2 = DRDB(in_ch=32)
        self.DRDB3 = DRDB(in_ch=32)
        self.DRDB4 = DRDB(in_ch=32)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv21 = nn.Conv2d(32, 1, 3, padding=1)

        self.relu = nn.PReLU()

        # self.skff = SKFF(64,2)
        # self.skff2 = SKFF(64, 2)
        self.ffm = FeatureFusionModule_ShowAttention(32)
        self.ffm2 = FeatureFusionModule(32)
        self.conv3 = nn.Conv2d(64, 32, 1, padding=0)
        self.conv4 = nn.Conv2d(128, 32, 1, padding=0)

    def forward(self, ir, vis, out1, out2):
        # print(np.shape(out1),'----------------')

        ir = ir[:, 0:1, :, :]
        x1 = self.conv1_ir(ir)
        x1 = self.relu(x1)
        x1 =self.DRDB1(x1)
        vis = vis[:, 0:1, :, :]
        x2 = self.conv1_vis(vis)
        x2 = self.relu(x2)
        x2 = self.DRDB2(x2)
        x1,x2,attentionlist = self.ffm(x1,x2,self.conv3(out1))
        x1 = self.DRDB3(x1)
        x2 = self.DRDB4(x2)
        x1, x2,_= self.ffm(x1, x2, self.conv4(out2))
        # f2 = self.skff2([f2,self.conv4(conv4)])
        f_final = self.relu(self.conv2((torch.cat([x1,x2],dim=1))))
        f_final= self.relu(self.conv21(f_final))

        return f_final, attentionlist

class Fusion_Network_rmseg(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_ir = nn.Conv2d(1, 64, 3, padding=1)
        self.conv1_vis = nn.Conv2d(1, 64, 3, padding=1)
        self.DRDB1 = DRDB(in_ch=64)
        self.DRDB2 = DRDB(in_ch=64)
        self.DRDB3 = DRDB(in_ch=64)
        self.DRDB4 = DRDB(in_ch=64)
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv21 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv22 = nn.Conv2d(32, 1, 3, padding=1)

        self.relu = nn.PReLU()


    def forward(self, ir, vis):
        # print(np.shape(out1),'----------------')

        ir = ir[:, 0:1, :, :]
        x1 = self.conv1_ir(ir)
        x1 = self.relu(x1)
        x1 =self.DRDB1(x1)
        vis = vis[:, 0:1, :, :]
        x2 = self.conv1_vis(vis)
        x2 = self.relu(x2)
        x2 = self.DRDB2(x2)
        # x1,x2 = self.ffm(x1,x2,self.conv3(out1))
        x1 = self.DRDB3(x1)
        x2 = self.DRDB4(x2)
        # x1, x2 = self.ffm(x1, x2, self.conv4(out2))
        # f2 = self.skff2([f2,self.conv4(conv4)])
        f_final = self.relu(self.conv2((torch.cat([x1,x2],dim=1))))
        f_final= self.relu(self.conv21(f_final))
        f_final= self.relu(self.conv22(f_final))

        # ones = torch.ones_like(f_final)
        # zeros = torch.zeros_like(f_final)
        # f_final = torch.where(f_final > ones, ones, f_final)
        # f_final = torch.where(f_final < zeros, zeros, f_final)
        # # new encode
        # f_final = (f_final - torch.min(f_final)) / (
        #         torch.max(f_final) - torch.min(f_final)
        # )
        return f_final

class Fusion_Network3_ac(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_ir = nn.Conv2d(1, 64, 3, padding=1)
        self.conv1_vis = nn.Conv2d(1, 64, 3, padding=1)
        self.DRDB1 = DRDB(in_ch=64)
        self.DRDB2 = DRDB(in_ch=64)
        self.DRDB3 = DRDB(in_ch=64)
        self.DRDB4 = DRDB(in_ch=64)
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1)
        # self.conv21 = nn.Conv2d(32, 1, 3, padding=1)

        self.relu = nn.PReLU()
        self.ffm = FeatureFusionModule(64)
        self.ffm2 = FeatureFusionModule(64)
        self.conv3 = nn.Conv2d(64, 64, 1, padding=0)
        self.conv4 = nn.Conv2d(128, 64, 1, padding=0)

        self.conv21 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv22 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, ir, vis, out1, out2):
        # print(np.shape(out1),'----------------')

        ir = ir[:, 0:1, :, :]
        x1 = self.conv1_ir(ir)
        x1 = self.relu(x1)
        x1 =self.DRDB1(x1)
        vis = vis[:, 0:1, :, :]
        x2 = self.conv1_vis(vis)
        x2 = self.relu(x2)
        x2 = self.DRDB2(x2)
        x1,x2 = self.ffm(x1,x2,self.conv3(out1))
        x1 = self.DRDB3(x1)
        x2 = self.DRDB4(x2)
        x1, x2 = self.ffm(x1, x2, self.conv4(out2))
        # f2 = self.skff2([f2,self.conv4(conv4)])
        f_final = self.relu(self.conv2((torch.cat([x1,x2],dim=1))))
        f_final= self.relu(self.conv21(f_final))
        f_final= self.relu(self.conv22(f_final))

        return f_final
class Network3(nn.Module):

    def __init__(self,backbone, num_classes=20, embedding_dim=256, pretrained=True):
        super(Network3, self).__init__()

        self.fusion_nums = 2
        self.seg_nums = 2
        self.fusion_channel = 48
        self.seg_channel = 64
        self.denoise_net = WeTr(backbone,num_classes,embedding_dim,pretrained)
        # self.discriminator = Discriminator()
        self.mean =[123.675, 116.28, 103.53]
        self.std = [58.395, 57.12, 57.375]
    def forward(self, fused_seg1):

        torch_norma = fused_seg1*255
        for index in range(3):
            torch_norma[:,index,:,:] = (torch_norma[:,index,:,:] - self.mean[index])/self.std[index]

        seg_map = self.denoise_net(torch_norma)
        return fused_seg1,fused_seg1,seg_map

    def _loss(self,fused_seg1,label,criterion):
        torch_norma = fused_seg1 * 255
        for index in range(3):
            torch_norma[:, index, :, :] = (torch_norma[:, index, :, :] - self.mean[index]) / self.std[index]
        seg_map = self.denoise_net(torch_norma)
        outputs = F.interpolate(seg_map, size=label.shape[1:], mode='bilinear', align_corners=False)
        denoise_loss = criterion(outputs,label.type(torch.long))
        return denoise_loss
    # def _loss(self, ir, vis,mask,labels):
    #     fused_img, seg1, seg2 = self(ir, vis)
    #     outputs = F.interpolate(seg2, size=labels.shape[1:], mode='bilinear', align_corners=False)
    #     denoise_loss = self.seg_loss(outputs,labels.type(torch.long))
    #     # vis = RGB2YCrCb(vis)
    #     return denoise_loss*1.25
    #
    #
    #
    #
    # def _detection_loss(self, ir, vis, labels):
    #     fused_img, seg_map = self(ir, vis)
    #
    #     outputs = F.interpolate(seg_map, size=labels.shape[1:], mode='bilinear', align_corners=False)
    #     # seg_loss = criterion(outputs, labels.type(torch.long))
    #     denoise_loss = self.seg_loss(outputs, labels.type(torch.long))        # vis = RGB2YCrCb(vis)
    #     return denoise_loss

    def enhance_net_parameters(self):
        return self.enhance_net.parameters()

    def denoise_net_parameters(self):
        return self.denoise_net.parameters()


# class Network_Totoal(nn.Module):
#
#     def __init__(self, segloss,backbone, num_classes=20, embedding_dim=256, pretrained=None):
#         super(Network_Totoal, self).__init__()
#         self.enhance_net = Fusion_Network2()
#         self.segmodel = Network3(segloss,backbone)
#         self.meanmodule = Mean()
#     def forward_seg(self, fused_seg1):
#         torch_norma = fused_seg1 * 255
#         for index in range(3):
#             torch_norma[:, index, :, :] = (torch_norma[:, index, :, :] - self.mean[index]) / self.std[index]
#
#         seg_map = self.segmodel.denoise_net(torch_norma)
#         return fused_seg1, fused_seg1, seg_map
#     def foward_fusion(self,ir,vis):
