import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as init


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])
   
        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')

        return out

def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2d):
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size)
            assert m.output_size == 1
            setattr(model, n, pool)



class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size)
        with torch.no_grad():
            self.forward(imgs)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

    
class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        self.SCA = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.SCA(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class LargeKernelAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class MSDABlock2(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., layernorm=True):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        
        self.conv2_d1 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=(3 - 1) * 1 // 2, stride=1, groups=dw_channel,
                               bias=True, dilation=1)
        self.conv2_d4 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=(3 - 1) * 4 // 2, stride=1, groups=dw_channel,
                               bias=True, dilation=4)
        self.conv2_d7 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=(3 - 1) * 7 // 2, stride=1, groups=dw_channel,
                               bias=True, dilation=7)
        
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.SCA = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.LKA = LargeKernelAttention(c)
        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.dwconv = nn.Conv2d(ffn_channel, ffn_channel, kernel_size=3, stride=1, padding=1, groups=ffn_channel)
       
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.conv2_d1(x) + self.conv2_d4(x) + self.conv2_d7(x) 
        
        x = self.sg(x)
        x_ca = x * self.SCA(x)
        x_sa = self.LKA(x)
        x = self.conv3(x_ca+x_sa)
        
        x = self.dropout1(x)

        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class MSLKB(nn.Module):
    def __init__(self, dim, k_size, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., layernorm=True):
        super().__init__()
        ker = k_size
        # print(k_size)
        pad = ker // 2
        dw_channel = dim * DW_Expand
        self.in_conv = nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
                    nn.GELU()
                    )
        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, stride=1, groups=dim,
                               bias=True)
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1,ker), padding=(0,pad), stride=1, groups=dim)
        self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(ker,1), padding=(pad,0), stride=1, groups=dim)
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)

        self.act = nn.GELU()


        # Simplified Channel Attention
        self.SCA = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * dim
        self.conv4 = nn.Conv2d(in_channels=dim, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        if layernorm:
            self.norm1 = LayerNorm2d(dim)
            self.norm2 = LayerNorm2d(dim)
        else:
            self.norm1 =  DynamicTanh(normalized_shape=dim, channels_last=False)
            self.norm2 =  DynamicTanh(normalized_shape=dim, channels_last=False)
 
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)
        x = self.in_conv(x)
        x = self.conv2(x)
        x = x + self.dw_13(x) + self.dw_31(x) + self.dw_33(x) + self.dw_11(x) + x*self.SCA(x)
        x = self.act(x)
        x = self.out_conv(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma



class MSDAB(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., layernorm=True):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        
        self.conv2_d1 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=(3 - 1) * 1 // 2, stride=1, groups=dw_channel,
                               bias=True, dilation=1)
        self.conv2_d4 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=(3 - 1) * 4 // 2, stride=1, groups=dw_channel,
                               bias=True, dilation=4)
        self.conv2_d7 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=(3 - 1) * 7 // 2, stride=1, groups=dw_channel,
                               bias=True, dilation=7)
        self.conv2_d9 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=(3 - 1) * 9 // 2, stride=1, groups=dw_channel,
                               bias=True, dilation=9)
        
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.SCA = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.LKA = LargeKernelAttention(c)
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.dwconv = nn.Conv2d(ffn_channel, ffn_channel, kernel_size=3, stride=1, padding=1, groups=ffn_channel)
        if layernorm:
            self.norm1 = LayerNorm2d(c)
            self.norm2 = LayerNorm2d(c)
        else:
            self.norm1 =  DynamicTanh(normalized_shape=c, channels_last=False)
            self.norm2 =  DynamicTanh(normalized_shape=c, channels_last=False)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.conv2_d1(x) + self.conv2_d4(x) + self.conv2_d7(x) + self.conv2_d9(x)
        
        x = self.sg(x)
        x_ca = x * self.SCA(x)
        x_sa = self.LKA(x)
        x = self.conv3(x_ca+x_sa)
        
        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma





class MSDAB2(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., layernorm=True):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        
        self.conv2_d1 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=(3 - 1) * 1 // 2, stride=1, groups=dw_channel,
                               bias=True, dilation=1)
        self.conv2_d4 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=(3 - 1) * 4 // 2, stride=1, groups=dw_channel,
                               bias=True, dilation=4)
        self.conv2_d9 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=(3 - 1) * 9 // 2, stride=1, groups=dw_channel,
                               bias=True, dilation=9)
        
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.SCA = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.LKA = LargeKernelAttention(c)
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        if layernorm:
            self.norm1 = LayerNorm2d(c)
            self.norm2 = LayerNorm2d(c)
        else:
            self.norm1 =  DynamicTanh(normalized_shape=c, channels_last=False)
            self.norm2 =  DynamicTanh(normalized_shape=c, channels_last=False)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.conv2_d1(x) + self.conv2_d4(x) + self.conv2_d9(x) + x
        
        x = self.sg(x)
        x_ca = x * self.SCA(x)
        x_sa = self.LKA(x)
        x = self.conv3(x_ca+x_sa)
        
        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"


class MZNet(nn.Module):

    def __init__(self, config, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()
        
        self.config = config
        img_channel = self.config.model.img_channel * 4
        width = self.config.model.width 
        enc_blk_nums = self.config.model.enc_blk_nums 
        middle_blk_num = self.config.model.middle_blk_num
        dec_blk_nums = self.config.model.dec_blk_nums
        layernorm = self.config.model.layernorm
        print(layernorm)
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        # FFSC
        self.combine_conv_level3 = nn.Conv2d(in_channels=480, out_channels=256, kernel_size=1)
        self.combine_conv_level2 = nn.Conv2d(in_channels=480, out_channels=128, kernel_size=1)
        self.combine_conv_level1 = nn.Conv2d(in_channels=480, out_channels=64, kernel_size=1)
        self.combine_conv_level0 = nn.Conv2d(in_channels=480, out_channels=32, kernel_size=1)
        self.combine_refine_level3 = NAFBlock(256)
        self.combine_refine_level2 = NAFBlock(128)
        self.combine_refine_level1 = NAFBlock(64)
        self.combine_refine_level0 = NAFBlock(32)
        
        
        self.decoder_out2 = nn.Conv2d(in_channels=128, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.decoder_out1 = nn.Conv2d(in_channels=64, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.reduce_chan_level3 = nn.Conv2d(in_channels=256 * 2, out_channels=256, kernel_size=1)
        self.reduce_chan_level2 = nn.Conv2d(in_channels=128 * 2, out_channels=128, kernel_size=1)
        self.reduce_chan_level1 = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1)
        self.reduce_chan_level0 = nn.Conv2d(in_channels=32 * 2, out_channels=32, kernel_size=1)
        self.final_refine = MSDAB2(32, layernorm=layernorm) if self.config.model.msdab_s else MSDAB(32, layernorm=layernorm)
        chan = width
        tip_block = (config.data.patch_size == 256)
        use_msdab2 = self.config.model.msdab_s

        # Encoder
        for i, num in enumerate(enc_blk_nums):
            if tip_block:
                block = MSDABlock2 if i == 3 else MSDAB
            else:
                block = MSDAB2 if use_msdab2 else MSDAB

            self.encoders.append(
                nn.Sequential(*[block(chan, layernorm=layernorm) for _ in range(num)])
            )
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2

        # Middle blocks
        self.middle_blks = nn.Sequential(
            *[MSLKB(chan, k_size=int(config.data.patch_size / 32 - 1), layernorm=layernorm)
            for _ in range(middle_blk_num)]
        )

        # Decoder
        for i, num in enumerate(dec_blk_nums):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan //= 2

            if tip_block:
                block = MSDABlock2 if i == 0 else MSDAB
            else:
                block = MSDAB2 if use_msdab2 else MSDAB

            self.decoders.append(
                nn.Sequential(*[block(chan, layernorm=layernorm) for _ in range(num)])
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        unshuffle_inp = torch.pixel_unshuffle(inp, 2)
        x = self.intro(unshuffle_inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)
        
        # FFSC
        combined_list=[]
        combined_3=torch.cat([F.interpolate(encs[0], scale_factor=0.125, mode='bilinear'),
                              F.interpolate(encs[1], scale_factor=0.25, mode='bilinear'), 
                              F.interpolate(encs[2], scale_factor=0.5, mode='bilinear'),
                              encs[3]], dim=1)
        combined_2=torch.cat([F.interpolate(encs[0], scale_factor=0.25, mode='bilinear'),
                              F.interpolate(encs[1], scale_factor=0.5, mode='bilinear'), 
                              encs[2],
                              F.interpolate(encs[3], scale_factor=2, mode='bilinear')], dim=1)
        combined_1=torch.cat([F.interpolate(encs[0], scale_factor=0.5, mode='bilinear'),
                              encs[1], 
                              F.interpolate(encs[2], scale_factor=2, mode='bilinear'),
                              F.interpolate(encs[3], scale_factor=4, mode='bilinear')], dim=1)
        combined_0=torch.cat([encs[0],
                              F.interpolate(encs[1], scale_factor=2, mode='bilinear'), 
                              F.interpolate(encs[2], scale_factor=4, mode='bilinear'),
                              F.interpolate(encs[3], scale_factor=8, mode='bilinear')], dim=1)

        combined_3 = self.combine_conv_level3(combined_3)
        combined_3 = self.combine_refine_level3(combined_3)
        combined_list.append(combined_3)
        combined_2 = self.combine_conv_level2(combined_2)
        combined_2 = self.combine_refine_level2(combined_2)
        combined_list.append(combined_2)
        combined_1 = self.combine_conv_level1(combined_1)
        combined_1 = self.combine_refine_level1(combined_1)
        combined_list.append(combined_1)
        combined_0 = self.combine_conv_level0(combined_0)
        combined_0 = self.combine_refine_level0(combined_0)
        combined_list.append(combined_0)
        

        for i, (decoder, up, skip) in enumerate(zip(self.decoders, self.ups, combined_list)):
            
            x = up(x)
            x = torch.cat([x, skip], dim=1)
        
            if i == 0:  
                x = self.reduce_chan_level3(x)
            elif i == 1:
                x = self.reduce_chan_level2(x)
            elif i == 2:
                x = self.reduce_chan_level1(x)
            else:
                x = self.reduce_chan_level0(x)
                
            x = decoder(x)
            if i == 1 or i == 2:
                if i == 1:
                    out2 = self.decoder_out2(x)
                    out2 = torch.pixel_shuffle(out2, 2)    
                else:
                    out3 = self.decoder_out1(x)
                    out3 = torch.pixel_shuffle(out3, 2)
                    
        x = self.final_refine(x)
        x = self.ending(x)
        x = torch.pixel_shuffle(x, 2)
        x = x + inp
        return out2, out3, x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class MZNetLocal(Local_Base, MZNet):
    def __init__(self, config, train_size=(1, 3, 512, 512), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        MZNet.__init__(self, config, **kwargs)
        train_size = (1, 3, config.data.patch_size, config.data.patch_size)
        N, C, H, W = train_size
        base_size = (int(H * config.model.tlc), int(W * config.model.tlc))

        self.eval()
        with torch.no_grad():
            print('convert local operation')
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

