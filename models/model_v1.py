import torch
import torch.nn as nn
from torchvision.ops.deform_conv import DeformConv2d
import torchvision
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
import torch.nn.functional as F
import math




def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv1x1(in_planes, out_planes, stride=1, has_bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


def conv1x1_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv1x1(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class BiFPN(nn.Module):
    def __init__(self, config, first_time):
        super(BiFPN, self).__init__()
        backbone = config.MODEL.BACKBONE 
        self.config = config
        self.train_input_size = config.TRAIN.SIZE

        if backbone in ['resnet18', 'resnet34']:
            nin = [64, 128, 256, 512]
        else:
            nin = [256, 512, 1024, 2048]

        ndim = 256
         
        self.first_time = first_time
        if self.first_time:
            self.fpn_in5 = conv3x3_bn_relu(nin[-1], ndim)
            self.fpn_in4 = conv3x3_bn_relu(nin[-2], ndim) 
            self.fpn_in3 = conv3x3_bn_relu(nin[-3], ndim)
            self.fpn_in2 = conv3x3_bn_relu(nin[-4], ndim)

        self.w_4_5 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w_3_4 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w_2_3 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w_2up_3in_3up = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.w_5in_5up = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w_3up_4in_4up = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)

        self.c5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.c4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.c3_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
 
        self.conv4_up = conv1x1(ndim, 256)  
        self.conv3_up = conv1x1(ndim, 256)  
        self.conv2_up = conv1x1(ndim, 256) 

        self.conv3_out = conv1x1(ndim, 256)  
        self.conv4_out = conv1x1(ndim, 256)  
        self.conv5_out = conv1x1(ndim, 256)  

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def swish(self, x):
        return x * x.sigmoid()

    def forward(self, inputs):
        if self.first_time:
            c2, c3, c4, c5 = inputs
            # stride:  4   8   16   32
            # channels:256 512 1024 2048
            #
            # c6_in=F.max_pool2d(self.extra_6(c5),kernel_size=3,stride=2,padding=1)
            # c7_in=F.max_pool2d(self.extra_7(c6_in),kernel_size=3,stride=2,padding=1)
            c5_in = self.fpn_in5(c5)
            c4_in = self.fpn_in4(c4)
            c3_in = self.fpn_in3(c3)
            c2_in = self.fpn_in2(c2)
        else:
            c2_in, c3_in, c4_in, c5_in = inputs  # ,c6_in,c7_in=inputs

        weights = F.relu(self.w_4_5)
        norm_weights = weights / (weights.sum() + 0.0001)
        # 10
        c4_up = self.conv4_up(self.swish(norm_weights[0] * c4_in + norm_weights[1] * self.c5_upsample(c5_in)))
        weights = F.relu(self.w_3_4)
        norm_weights = weights / (weights.sum() + 0.0001)
        # 11
        c3_up = self.conv3_up(self.swish(norm_weights[0] * c3_in + norm_weights[1] * self.c4_upsample(c4_up)))
        weights = F.relu(self.w_2_3)
        norm_weights = weights / (weights.sum() + 0.0001)
        # 12
        c2_out = self.conv2_up(self.swish(norm_weights[0] * c2_in + norm_weights[1] * self.c3_upsample(c3_up)))
        weights = F.relu(self.w_2up_3in_3up)
        norm_weights = weights / (weights.sum() + 0.0001)
        # 13
        c3_out = self.conv3_out(self.swish(
            norm_weights[0] * c3_in + norm_weights[1] * c3_up + norm_weights[2] * F.max_pool2d(c2_out, kernel_size=3,
                                                                                               stride=2, padding=1)))
        weights = F.relu(self.w_3up_4in_4up)
        norm_weights = weights / (weights.sum() + 0.0001)
        # 14
        c4_out = self.conv4_out(self.swish(
            norm_weights[0] * c4_in + norm_weights[1] * c4_up + norm_weights[2] * F.max_pool2d(c3_out, kernel_size=3,
                                                                                               stride=2, padding=1)))
        weights = F.relu(self.w_5in_5up)
        norm_weights = weights / (weights.sum() + 0.0001)
        # 15
        c5_out = self.conv5_out(self.swish(
            norm_weights[0] * c5_in + norm_weights[1] * F.max_pool2d(c4_out, kernel_size=3,
                                                                     stride=2, padding=1)))

        return c2_out, c3_out, c4_out, c5_out


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class Spotting_Head(nn.Module):
    def __init__(self, config, converter):
        super(Spotting_Head, self).__init__()
        self.config = config
        self.converter = converter
        ndim = 192
        self.train_input_size = config.TRAIN.SIZE
        self.cls_tail = nn.Sequential(
            conv3x3_bn_relu(ndim, 32, 1),
            conv1x1(32, 1, 1, True),
            nn.Sigmoid()
        )

        self.loc_tail = nn.Sequential(
            conv3x3_bn_relu(ndim, 32, 1),
            conv1x1(32, 4, 1, True),
            nn.Sigmoid()
        )

        self.weight_tail = nn.Sequential(
            conv3x3_bn_relu(ndim, 32, 1),
            conv1x1(32, config.MODEL.WEIGHTS_NUM, 1, True),
            nn.Tanh()
        )

        self.mask_tail_1 = conv3x3_bn_relu(ndim, 32, 1)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.mask_tail_2 = nn.Sequential(
            conv1x1(32, config.MODEL.WEIGHTS_NUM, 1, True),
            nn.ReLU(inplace=True)
        )

        self.offsets = conv1x1(4, 18 * 4, 1, True)
        self.relu = nn.ReLU(inplace=True)
        self.shape_adaption = DeformConv2d(in_channels=ndim, out_channels=ndim, kernel_size=3, stride=1, padding=1,
                                           groups=4)

        self.sampler = nn.Sequential(
            conv3x3_bn_relu(ndim, 192, 1),
            conv3x3_bn_relu(192, 128, 1),
            conv3x3_bn_relu(128, 64, 1),
            conv1x1(64, config.MODEL.SAMPLE_POINT_NUMS * 2, 1, True),
        )
        self.rec_proj = nn.Sequential(
            conv3x3_bn_relu(ndim, 128, 1),
            conv3x3_bn_relu(128, 128, 1),
            conv3x3_bn_relu(128, ndim, 1),
            conv3x3_bn_relu(ndim, ndim, 1),
        )

        self.fpn_strides = [4, 8, 16, 32]
        #self.converter = StrLabelConverter(config.alphabet, config.ignore_case, config.max_text_length)
        self.rec_tail = nn.Conv2d(ndim, self.converter.nClasses, kernel_size=1, stride=1, bias=True)
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in self.fpn_strides])
        self.scales_off = nn.ModuleList([Scale(init_value=1.0) for _ in self.fpn_strides])
        self.cls_tail.apply(self.weights_init)
        self.loc_tail.apply(self.weights_init)
        self.weight_tail.apply(self.weights_init)
        self.mask_tail_1.apply(self.weights_init)
        self.mask_tail_2.apply(self.weights_init)
        self.sampler.apply(self.weights_init)
        self.rec_proj.apply(self.weights_init)
        self.rec_tail.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def forward(self, feats):
        score_preds = []
        loc_preds = []
        weight_preds = []
        mask_preds = []
        rec_preds = []
        sampler_preds = []
        for l, feature in enumerate(feats):
            score_pred, loc_pred, weight_pred, mask_pred, rec_pred, sampler_pred = None, None, None, None, None, None
            score_pred = self.cls_tail(feature)
            loc_pred = self.scales[l](self.loc_tail(feature)) * self.train_input_size
            weight_pred = self.weight_tail(feature)
            mask_pred = self.mask_tail_1(feature)
            if l == 0:
                mask_pred = self.mask_tail_2(self.upsample_4(mask_pred))
            else:
                mask_pred = self.mask_tail_2(self.upsample_8(mask_pred))
            score_preds.append(score_pred)
            loc_preds.append(loc_pred)
            weight_preds.append(weight_pred)
            mask_preds.append(mask_pred)
            sampler_pred = self.sampler(feature)
            proj_fuse = self.rec_proj(feature)
            rec_pred = self.rec_tail(proj_fuse)
            sampler_preds.append(sampler_pred)
            rec_preds.append(rec_pred)

        return score_preds, loc_preds, weight_preds, mask_preds, rec_preds, sampler_preds


class SRSTS_v1(nn.Module):
    def __init__(self, config, converter):
        super(SRSTS_v1, self).__init__()
        backbone = config.MODEL.BACKBONE
        self.config = config
        self.train_input_size = config.TRAIN.SIZE
        self.backbone = eval(backbone)(pretrained=True)
        self.repeated_bifpn = nn.ModuleList([BiFPN(config, first_time=1)  ,BiFPN(config, first_time=0)])
        ndim = 256
        self.fpn_out5_3 = nn.Sequential(
            conv3x3_bn_relu(ndim, 64),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))

        self.fpn_out4_3 = nn.Sequential(
            conv3x3_bn_relu(ndim, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

        self.fpn_out4_2 = nn.Sequential(
            conv3x3_bn_relu(ndim, 64),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))

        self.fpn_out3_3 = nn.Sequential(
            conv3x3_bn_relu(ndim, 64))

        self.fpn_out3_2 = nn.Sequential(
            conv3x3_bn_relu(ndim, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

        self.fpn_out2_2 = conv3x3_bn_relu(ndim, 64)
        self.e2e_head = Spotting_Head(config, converter)

    def forward(self, imgs):
        feats = self.backbone(imgs)
        for bifpn in self.repeated_bifpn:
            feats = bifpn(feats)
        p5_3 = self.fpn_out5_3(feats[3])
        p4_3 = self.fpn_out4_3(feats[2])
        p3_3 = self.fpn_out3_3(feats[1])

        p4_2 = self.fpn_out4_2(feats[2])
        p3_2 = self.fpn_out3_2(feats[1])
        p2_2 = self.fpn_out2_2(feats[0])
        features = [torch.cat((p4_2, p3_2, p2_2), 1), torch.cat((p5_3, p4_3, p3_3), 1)]
        score_preds, loc_preds, weight_preds, mask_preds, rec_preds, sampler_preds = self.e2e_head(features)
        return score_preds, loc_preds, weight_preds, mask_preds, rec_preds, sampler_preds
