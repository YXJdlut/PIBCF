import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange

class FusionNet(nn.Module):
    def __init__(self,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias'):
        super(FusionNet, self).__init__()

        # 编码器
        self.encoder = Restormer_Encoder(
            dim=dim,
            num_blocks=num_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type
        )

        # 融合解码器
        self.decoder = Restormer_Decoder(
            dim=dim,
            num_blocks=num_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type
        )

        # 单模态 CT 解码器
        self.decoder_ct = Restormer_Decoder_Single(
            in_channels=dim,
            out_channels=1,
            num_blocks=2,
            heads=heads[0],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type
        )

        # 单模态 MRI 解码器
        self.decoder_mri = Restormer_Decoder_Single(
            in_channels=dim,
            out_channels=1,
            num_blocks=2,
            heads=heads[0],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type
        )

    # def forward(self, ct_img, mri_img):
    #     # 编码器提取特征
    #     fused_feats, ct_feat, mri_feat, _, _ = self.encoder(ct_img, mri_img)

    #     # 融合解码 -> 最终融合图像
    #     fused_img, _ = self.decoder(fused_feats)

    #     # 单模态解码
    #     ct_decoded = self.decoder_ct(ct_feat)
    #     mri_decoded = self.decoder_mri(mri_feat)

    #     # ✅ 只返回训练中需要的 5 个值
    #     # return fused_img, ct_decoded, mri_decoded, ct_feat, mri_feat
    #     return fused_img, None, None, None, None
    def forward(self, ct_img, mri_img):
        # 编码器提取特征
        fused_feats, ct_feat, mri_feat, _, _ = self.encoder(ct_img, mri_img)

        # 融合解码 -> 最终融合图像 (传入 ct/mri 的均值作为 inp_img 做残差)
        inp_img = 0.7 * ct_img + 0.3 * mri_img   # 🔑 使用均值作为参考
        # inp_img = ct_img
        fused_img, _ = self.decoder(fused_feats, inp_img=inp_img)

        # 单模态解码
        ct_decoded = self.decoder_ct(ct_feat)
        mri_decoded = self.decoder_mri(mri_feat)

        return fused_img, None, None, None, None




def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)



class AttentionBase(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.proj(out)
        return out
    
class CotLayer(nn.Module):
    def __init__(self, dim):
        super(CotLayer, self).__init__()
        self.key_embed = Convlutioanl(dim, dim)

        factor = 8
        self.embed = nn.Sequential(
            nn.Conv2d(dim, dim // factor, 1, bias=False),
            nn.BatchNorm2d(dim // factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // factor, 1, kernel_size=1),
            nn.BatchNorm2d(1)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(dim)
        )

    def forward(self, v, k, q):
        k = self.key_embed(k)
        qk = q + k
        w = self.embed(qk)
        v = self.conv1x1(v)
        mul = w * v
        out = mul + k
        return out
   
class Convlutioanl(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Convlutioanl, self).__init__()
        self.padding = (1, 1, 1, 1)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=0, stride=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = F.pad(x, self.padding, mode='replicate')
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    
class Mlp(nn.Module):
    def __init__(self, in_features, ffn_expansion_factor=2, bias=False):
        super().__init__()
        hidden_features = int(in_features * ffn_expansion_factor)
        self.project_in = nn.Conv2d(in_features, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, in_features, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return rearrange(self.body(rearrange(x, 'b c h w -> b (h w) c')), 'b (h w) c -> b c h w', h=h, w=w)

class CTFeatureExtractionBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.0,
                 qkv_bias=False):
        super(CTFeatureExtractionBlock, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias)

        self.local_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),  # Depthwise
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1)  # Pointwise
        )

        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim, ffn_expansion_factor=ffn_expansion_factor)

    def forward(self, x):
        global_feat = self.attn(self.norm1(x))
        local_feat = self.local_conv(x)
        x = x + global_feat + local_feat
        x = x + self.mlp(self.norm2(x))
        return x

class CTFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.0,
                 qkv_bias=False,
                 num_layers=3):
        super(CTFeatureExtraction, self).__init__()
        self.blocks = nn.Sequential(*[
            CTFeatureExtractionBlock(dim=dim, num_heads=num_heads,
                                      ffn_expansion_factor=ffn_expansion_factor,
                                      qkv_bias=qkv_bias)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        return self.blocks(x)

class MRIFeatureExtractionBlock(nn.Module):
    def __init__(self,
                 dim,
                 ffn_expansion_factor=1.0):
        super(MRIFeatureExtractionBlock, self).__init__()
        self.local_path = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        )
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.norm = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim, ffn_expansion_factor=ffn_expansion_factor)

    def forward(self, x):
        local_feat = self.local_path(x)
        scale = self.attn(local_feat)
        x = x + local_feat * scale
        x = x + self.mlp(self.norm(x))
        return x

class MRIFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,
                 ffn_expansion_factor=1.0,
                 num_layers=3):
        super(MRIFeatureExtraction, self).__init__()
        self.blocks = nn.Sequential(*[
            MRIFeatureExtractionBlock(dim=dim, ffn_expansion_factor=ffn_expansion_factor)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        return self.blocks(x)

class DynamicCrossInteraction(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 2, kernel_size=1)
        )

    def forward(self, ct_feat, mri_feat):
        fused_input = torch.cat([ct_feat, mri_feat], dim=1)  # [B, 2C, H, W]
        weights = self.attn_conv(fused_input)                # [B, 2, H, W]
        weights = F.softmax(weights, dim=1)                  # w_ct + w_mri = 1

        w_ct = weights[:, 0:1, :, :]
        w_mri = weights[:, 1:2, :, :]

        fused = w_ct * ct_feat + w_mri * mri_feat
        return fused


class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )
    def forward(self, x):
        return self.bottleneckBlock(x)

class DetailNode(nn.Module):
    def __init__(self):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.shffleconv = nn.Conv2d(64, 64, kernel_size=1,
                                    stride=1, padding=0, bias=True)
    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2
    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2

class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=3):
        super(DetailFeatureExtraction, self).__init__()
        INNmodules = [DetailNode() for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)
    def forward(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)

# =============================================================================

# =============================================================================
import numbers
##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x



class Restormer_Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias'):

        super(Restormer_Encoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[0])
        ])

        self.ctFeature = CTFeatureExtraction(dim=dim, num_heads=heads[2],
                                             ffn_expansion_factor=ffn_expansion_factor,
                                             qkv_bias=bias, num_layers=3)
        self.mriFeature = MRIFeatureExtraction(dim=dim,
                                               ffn_expansion_factor=ffn_expansion_factor,
                                               num_layers=3)

        # ✅ 使用新的动态融合交互模块
        self.interaction_blocks = nn.ModuleList([
            DynamicCrossInteraction(dim) for _ in range(3)
        ])

        # ❌ 移除 FusionModule（不再使用）
        # self.fusion_module = FusionModule(dim)

        # 重建输出
        self.reconstructor_ct = nn.Conv2d(dim, 1, kernel_size=1)
        self.reconstructor_mri = nn.Conv2d(dim, 1, kernel_size=1)

    def forward(self, ct_img, mri_img):
        ct_embed = self.patch_embed(ct_img)
        mri_embed = self.patch_embed(mri_img)

        ct_feat = self.encoder_level1(ct_embed)
        mri_feat = self.encoder_level1(mri_embed)

        x_ct = ct_feat
        x_mri = mri_feat

        fused_feats = []

        for i in range(3):
            x_ct = self.ctFeature.blocks[i](x_ct)
            x_mri = self.mriFeature.blocks[i](x_mri)
            fused_i = self.interaction_blocks[i](x_ct, x_mri)  # ✅ 新融合方式
            fused_feats.append(fused_i)

        recon_ct = self.reconstructor_ct(x_ct)
        recon_mri = self.reconstructor_mri(x_mri)

        return fused_feats, ct_feat, mri_feat, recon_ct, recon_mri

class Restormer_Decoder_Single(nn.Module):
    def __init__(self,
                 in_channels=64,
                 out_channels=1,
                 num_blocks=2,
                 heads=8,
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias'):
        super(Restormer_Decoder_Single, self).__init__()

        self.transformer = nn.Sequential(*[
            TransformerBlock(
                dim=in_channels,
                num_heads=heads,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks)
        ])

        self.output = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1, bias=bias),
            nn.Sigmoid()

        )

    def forward(self, x):
        x = self.transformer(x)
        return self.output(x)



#修改前
# class Restormer_Decoder(nn.Module):
#     def __init__(self,
#                  out_channels=1,
#                  dim=64,
#                  num_blocks=[4, 4],
#                  heads=[8, 8, 8],
#                  ffn_expansion_factor=2,
#                  bias=False,
#                  LayerNorm_type='WithBias'):
#         super(Restormer_Decoder, self).__init__()

#         # 多层特征融合
#         self.fuse = nn.Sequential(
#             nn.Conv2d(dim * 3, dim, kernel_size=1, bias=bias),
#             nn.ReLU(inplace=True)
#         )

#         self.reduce_channel = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

#         self.transformer = nn.Sequential(*[
#             TransformerBlock(
#                 dim=dim,
#                 num_heads=heads[1],
#                 ffn_expansion_factor=ffn_expansion_factor,
#                 bias=bias,
#                 LayerNorm_type=LayerNorm_type
#             ) for _ in range(num_blocks[1])
#         ])

#         self.output = nn.Sequential(
#             nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1, bias=bias),
#             nn.LeakyReLU(),
#             nn.Conv2d(dim // 2, out_channels, kernel_size=3, padding=1, bias=bias),
           

#         )

#     def forward(self, fused_feats):  # 输入: List[Tensor]，3个融合特征
#         # fused_feats = [fused1, fused2, fused3]
#         fused_cat = torch.cat(fused_feats, dim=1)  # 通道拼接
#         x = self.fuse(fused_cat)
#         x = self.reduce_channel(x)
#         x = self.transformer(x)
#         out = self.output(x)
#         return torch.sigmoid(out), x  # out: 融合图像; x: 最后一层特征
class Restormer_Decoder(nn.Module):
    def __init__(self,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias'):
        super(Restormer_Decoder, self).__init__()

        # 多层特征融合
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1, bias=bias),
            nn.ReLU(inplace=True)
        )

        self.reduce_channel = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.transformer = nn.Sequential(*[
            TransformerBlock(
                dim=dim,
                num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[1])
        ])

        self.output = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(dim // 2, out_channels, kernel_size=3, padding=1, bias=bias),
        )
        self.sigmoid = nn.Sigmoid()

    # def forward(self, fused_feats, inp_img=None):  
    #     # fused_feats = [fused1, fused2, fused3]
    #     fused_cat = torch.cat(fused_feats, dim=1)  # 通道拼接
    #     x = self.fuse(fused_cat)
    #     x = self.reduce_channel(x)
    #     x = self.transformer(x)
    #     out = self.output(x)

    #     # 🔑 残差连接（和 CDDFuse 对齐）
    #     if inp_img is not None:
    #         out = out + inp_img  

    #     return self.sigmoid(out), x
    def forward(self, fused_feats, inp_img=None):  
        # fused_feats = [fused1, fused2, fused3]
        fused_cat = torch.cat(fused_feats, dim=1)  # 通道拼接
        x = self.fuse(fused_cat)
        x = self.reduce_channel(x)
        x = self.transformer(x)
        out = self.output(x)

        # 🔑 残差连接（和 CDDFuse 对齐）
        if inp_img is not None:
            out = out + inp_img  

        # 🔑 CDDFuse 风格的动态归一化
        out = (out - out.amin(dim=(1,2,3), keepdim=True)) / (
            out.amax(dim=(1,2,3), keepdim=True) - out.amin(dim=(1,2,3), keepdim=True) + 1e-8
        )

        return out, x







    
if __name__ == '__main__':
    height = 256
    width = 256
    window_size = 8
    modelE = Restormer_Encoder().cuda()
    modelD = Restormer_Decoder().cuda()

