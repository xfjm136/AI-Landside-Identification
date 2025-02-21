import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.backbone.xception import xception
from nets.backbone.mobilenetv2 import mobilenetv2
from nets.backbone.mobilenetv3 import mobilenet_v3
from nets.backbone.mobilenet.mobilenetv4 import mobilenet_v4
from efficientnet_pytorch import EfficientNet
from nets.backbone.microsoft_swintransformer import SwinTransformer
import torch.nn as nn
from torchvision.models import resnet50,ResNet50_Weights,VGG16_Weights,vgg16
from torchvision.models import efficientnet_b0
from torchvision.models.efficientnet import EfficientNet_B0_Weights
from functools import partial
from nets.Attention.ALL import AttentionFactory

class ResNet(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(ResNet, self).__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None

        model = resnet50(weights=weights)
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4
        )

        self.down_idx = [4, 5, 6, 7]  # layer1, layer2, layer3, layer4

        if downsample_factor == 8:
            for i in range(6, 7):  # Adjust layer3
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(7, 8):  # Adjust layer4
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:5](x)  # layer1 output
        x = self.features[5:](low_level_features)  # layer2 to layer4
        return low_level_features, x

class VGG(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(VGG, self).__init__()

        # 根据 pretrained 动态设置 weights 参数
        weights = VGG16_Weights.DEFAULT if pretrained else None

        model = vgg16(weights=weights)
        self.features = model.features

        self.total_idx = len(self.features)
        self.down_idx = [4, 9, 16]  # layer1, layer2, layer3

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):  # Adjust layer3
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):  # Adjust layer4
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:5](x)  # First block
        x = self.features[5:](low_level_features)  # Remaining blocks
        return low_level_features, x

class EfficientNet(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(EfficientNet, self).__init__()

        # 根据 pretrained 动态设置 weights 参数
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None

        model = efficientnet_b0(weights=weights)
        self.features = model.features

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 6]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))

    def _nostride_dilate(self, m, dilate):
        # 只在 Conv2d 层修改 stride 和 dilation
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:3](x)
        x = self.features[3:](low_level_features)
        return low_level_features, x
class SwinTransformerBackbone(nn.Module):
    def __init__(self, model_name='swin_tiny_patch4_window7_224', pretrained=True, downsample_factor=16):
        super(SwinTransformerBackbone, self).__init__()
        self.model = SwinTransformer(model_name=model_name, pretrained=pretrained)
        self.downsample_factor = downsample_factor

        # 根据downsample_factor设置低层和高层特征的索引
        if downsample_factor == 8:
            self.low_level_layer_idx = 0  # stage1 (index 0)
            self.high_level_layer_idx = 2  # stage3 (index 2)
        elif downsample_factor == 16:
            self.low_level_layer_idx = 0  # stage1 (index 0)
            self.high_level_layer_idx = 2  # stage3 (index 2)
        else:
            raise ValueError('Unsupported downsample factor - `{}`, Use 8 or 16.'.format(downsample_factor))

    def forward(self, x):
        # 获取特征图，返回的是一个元组
        features = self.model.forward_features(x)

        # 获取低层特征（stage1），使用索引访问
        low_level_features = features[0]  # stage1 对应 features[0]

        # 获取高层特征（stage3），使用索引访问
        high_level_features = features[1]  # stage3 对应 features[1]

        return low_level_features, high_level_features
class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x

class MobileNetV3(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV3, self).__init__()
        from functools import partial

        model = mobilenet_v3(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        stage1 = self.features[0:2](x)
        stage2 = self.features[2:3](stage1)
        stage3= self.features[3:6](stage2)
        stage4= self.features[6:10](stage3)
        stage5= self.features[10:12](stage4)
        x= self.features[12:15](stage5)

        return stage2,x
# if __name__ == '__main__':
#     model = MobileNetV3(downsample_factor=4, pretrained=False)
#     input = torch.randn(1, 3, 512, 512)
#     output = model(input)
#     print(output[0].shape, output[1].shape, output[2].shape, output[3].shape)

class mobilenetv4(nn.Module):
    def __init__(self, model='MobileNetV4ConvSmall',pretrained=False):
        super().__init__()

        # Initialize MobileNetV4 models for low and high level features
        self.low_level_features = mobilenet_v4(model=model,pretrained=pretrained)
        self.high = mobilenet_v4(model=model,pretrained=pretrained)

    def forward(self, x):
        # 获取 MobileNetV4 中的第二层 (layer2) 和第四层 (layer4) 特征
        features = self.low_level_features(x)

        # 提取第二层和第四层的特征
        low_level_features = features[0]  # 第二层是 layer2
        high_level_features = features[3]  # 第四层是 layer4

        return low_level_features, high_level_features
# if __name__ == '__main__':
#     model = mobilenetv4('MobileNetV4ConvSmall')
#     x = torch.randn(1, 3, 512, 512)
#     y = model(x)
#     for i in range(len(y)):
#         print(y[i].shape)

class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.conv_cat1 = nn.Sequential(
            nn.Conv2d(dim_out * 2, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)

        conv3x3_2=torch.cat([conv3x3_2,conv3x3_3],dim=1)
        conv3x3_2=self.conv_cat1(conv3x3_2)

        conv3x3_1=torch.cat([conv3x3_1,conv3x3_2],dim=1)
        conv3x3_1=self.conv_cat1(conv3x3_1)


        # -----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        # -----------------------------------------#
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        # -----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        # -----------------------------------------#
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilenetv3", pretrained=True, downsample_factor=16,attention_type=None):
        super(DeepLab, self).__init__()

        if backbone == "xception":
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif "mobilenetv2"in backbone:
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        elif "mobilenetv3"in backbone:
            self.backbone = MobileNetV3(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 160#([1, 160, 16, 16])

            low_level_channels = 24#([1, 24, 128, 128])

        elif 'efficientnet_b0'in backbone:
            self.backbone = EfficientNet( pretrained=pretrained, downsample_factor=downsample_factor)
            in_channels = 1280
            low_level_channels = 24
        elif "swintransformer" in backbone:
            self.backbone = SwinTransformerBackbone(model_name=backbone, pretrained=pretrained,
                                                    downsample_factor=downsample_factor)
            in_channels = 768  # 根据模型的高特征层输出维度设置
            low_level_channels = 192  # 根据模型的低特征层输出维度设置

        elif  "vgg16"in backbone:
            self.backbone = VGG(pretrained=pretrained, downsample_factor=downsample_factor)
            in_channels = 512  # VGG16高特征层的输出通道数
            low_level_channels = 64  # VGG16低特征层的输出通道数
        elif "resnet50"in backbone:
            self.backbone = ResNet(pretrained=pretrained, downsample_factor=downsample_factor)
            in_channels = 2048  # resnet50高特征层的输出通道数
            low_level_channels = 256  # resnet50低特征层的输出通道数
        elif "mobilenetv4"in backbone:
            self.backbone = mobilenet_v4(pretrained=pretrained)
            in_channels = 1280  # MobileNetV4高特征层的输出通道数
            low_level_channels = 64  # MobileNetV4低特征层的输出通道数
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        self.attention_low = AttentionFactory.get_attention(attention_type, low_level_channels)
        self.attention_high = AttentionFactory.get_attention(attention_type, in_channels)

        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16 // downsample_factor)

        self.cls_conv = nn.Conv2d(64, num_classes, 1, stride=1)


        # ----------------------------------#
        #   浅层特征边
        # ----------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)


    def forward(self, x):
        H, W = x.size(2), x.size(3)
        low_level_features,x = self.backbone(x)
        low_level_features = self.attention_low(low_level_features)
        x = self.attention_high(x)

        x = self.aspp(x)

        low_level_features = self.shortcut_conv(low_level_features)
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear',
                          align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

