from collections import OrderedDict

import torch
import torch.nn as nn

from nets.mobilenet_v1 import mobilenet_v1
from nets.mobilenet_v2 import mobilenet_v2
from nets.mobilenet_v3 import mobilenet_v3
from nets.ghostnet import ghostnet

class MobileNetV1(nn.Module):
    def __init__(self, pretrained = False):
        super(MobileNetV1, self).__init__()
        self.model = mobilenet_v1(pretrained=pretrained)

    def forward(self, x):
        out3 = self.model.stage1(x)
        out4 = self.model.stage2(out3)
        out5 = self.model.stage3(out4)
        return out3, out4, out5

class MobileNetV2(nn.Module):
    def __init__(self, pretrained = False):
        super(MobileNetV2, self).__init__()
        self.model = mobilenet_v2(pretrained=pretrained)

    def forward(self, x):
        out3 = self.model.features[:7](x)
        out4 = self.model.features[7:14](out3)
        out5 = self.model.features[14:18](out4)
        return out3, out4, out5

class MobileNetV3(nn.Module):
    def __init__(self, pretrained = False):
        super(MobileNetV3, self).__init__()
        self.model = mobilenet_v3(pretrained=pretrained)

    def forward(self, x):
        out3 = self.model.features[:7](x)
        out4 = self.model.features[7:13](out3)
        out5 = self.model.features[13:16](out4)
        return out3, out4, out5

class GhostNet(nn.Module):
    def __init__(self, pretrained=True):
        super(GhostNet, self).__init__()
        model = ghostnet()
        if pretrained:
            state_dict = torch.load("model_data/ghostnet_weights.pth")
            model.load_state_dict(state_dict)
        del model.global_pool
        del model.conv_head
        del model.act2
        del model.classifier
        del model.blocks[9]
        self.model = model
        self.layers_out_filters = [16, 24, 40, 112, 160]

    def forward(self, x):
        x = self.model.conv_stem(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        feature_maps = []

        for idx, block in enumerate(self.model.blocks):
            x = block(x)
            if idx in [2,4,6,8]:
                feature_maps.append(x)
        return feature_maps[1:]

def conv2d(filter_in, filter_out, kernel_size, groups=1, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, groups=groups, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.ReLU6(inplace=True)),
    ]))

def conv_dw(filter_in, filter_out, stride = 1):
    return nn.Sequential(
        nn.Conv2d(filter_in, filter_in, 3, stride, 1, groups=filter_in, bias=False),
        nn.BatchNorm2d(filter_in),
        nn.ReLU6(inplace=True),

        nn.Conv2d(filter_in, filter_out, 1, 1, 0, bias=False),
        nn.BatchNorm2d(filter_out),
        nn.ReLU6(inplace=True),
    )

#---------------------------------------------------#
#   SPP结构，利用不同大小的池化核进行池化
#   池化后堆叠
#---------------------------------------------------#
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features

#---------------------------------------------------#
#   卷积 + 上采样
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

#---------------------------------------------------#
#   三次卷积块
#---------------------------------------------------#
def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

#---------------------------------------------------#
#   五次卷积块
#---------------------------------------------------#
def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        conv2d(filters_list[1], filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

#---------------------------------------------------#
#   最后获得yolov4的输出
#---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv_dw(in_filters, filters_list[0]),
        
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m

    
#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, backbone="mobilenetv2", pretrained=False):
        super(YoloBody, self).__init__()
        #---------------------------------------------------#   
        #   生成mobilnet的主干模型，获得三个有效特征层。
        #---------------------------------------------------#
        if backbone == "mobilenetv1":
            #---------------------------------------------------#   
            #   52,52,256；26,26,512；13,13,1024
            #---------------------------------------------------#
            self.backbone   = MobileNetV1(pretrained=pretrained)
            in_filters      = [256,512,1024]
        elif backbone == "mobilenetv2":
            #---------------------------------------------------#   
            #   52,52,32；26,26,92；13,13,320
            #---------------------------------------------------#
            self.backbone   = MobileNetV2(pretrained=pretrained)
            in_filters      = [32,96,320]
        elif backbone == "mobilenetv3":
            #---------------------------------------------------#   
            #   52,52,40；26,26,112；13,13,160
            #---------------------------------------------------#
            self.backbone   = MobileNetV3(pretrained=pretrained)
            in_filters      = [40,112,160]
        elif backbone == "ghostnet":
            #---------------------------------------------------#   
            #   52,52,40；26,26,112；13,13,160
            #---------------------------------------------------#
            self.backbone   = GhostNet(pretrained=pretrained)
            in_filters      = [40,112,160]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenetv1, mobilenetv2, mobilenetv3, ghostnet.'.format(backbone))


        self.conv1           = make_three_conv([512, 1024], in_filters[2])
        self.SPP             = SpatialPyramidPooling()
        self.conv2           = make_three_conv([512, 1024], 2048)

        self.upsample1       = Upsample(512, 256)
        self.conv_for_P4     = conv2d(in_filters[1], 256,1)
        self.make_five_conv1 = make_five_conv([256, 512], 512)

        self.upsample2       = Upsample(256, 128)
        self.conv_for_P3     = conv2d(in_filters[0], 128,1)
        self.make_five_conv2 = make_five_conv([128, 256], 256)

        # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
        self.yolo_head3      = yolo_head([256, len(anchors_mask[0]) * (5 + num_classes)], 128)

        self.down_sample1    = conv_dw(128, 256, stride = 2)
        self.make_five_conv3 = make_five_conv([256, 512], 512)

        # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
        self.yolo_head2      = yolo_head([512, len(anchors_mask[1]) * (5 + num_classes)], 256)

        self.down_sample2    = conv_dw(256, 512, stride = 2)
        self.make_five_conv4 = make_five_conv([512, 1024], 1024)

        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        self.yolo_head1      = yolo_head([1024, len(anchors_mask[2]) * (5 + num_classes)], 512)


    def forward(self, x):
        #  backbone
        x2, x1, x0 = self.backbone(x)

        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,2048 
        P5 = self.conv1(x0)
        P5 = self.SPP(P5)
        # 13,13,2048 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        P5 = self.conv2(P5)

        # 13,13,512 -> 13,13,256 -> 26,26,256
        P5_upsample = self.upsample1(P5)
        # 26,26,512 -> 26,26,256
        P4 = self.conv_for_P4(x1)
        # 26,26,256 + 26,26,256 -> 26,26,512
        P4 = torch.cat([P4,P5_upsample],axis=1)
        # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        P4 = self.make_five_conv1(P4)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        P4_upsample = self.upsample2(P4)
        # 52,52,256 -> 52,52,128
        P3 = self.conv_for_P3(x2)
        # 52,52,128 + 52,52,128 -> 52,52,256
        P3 = torch.cat([P3,P4_upsample],axis=1)
        # 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        P3 = self.make_five_conv2(P3)

        # 52,52,128 -> 26,26,256
        P3_downsample = self.down_sample1(P3)
        # 26,26,256 + 26,26,256 -> 26,26,512
        P4 = torch.cat([P3_downsample,P4],axis=1)
        # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        P4 = self.make_five_conv3(P4)

        # 26,26,256 -> 13,13,512
        P4_downsample = self.down_sample2(P4)
        # 13,13,512 + 13,13,512 -> 13,13,1024
        P5 = torch.cat([P4_downsample,P5],axis=1)
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        P5 = self.make_five_conv4(P5)

        #---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size,75,52,52)
        #---------------------------------------------------#
        out2 = self.yolo_head3(P3)
        #---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size,75,26,26)
        #---------------------------------------------------#
        out1 = self.yolo_head2(P4)
        #---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size,75,13,13)
        #---------------------------------------------------#
        out0 = self.yolo_head1(P5)

        return out0, out1, out2

