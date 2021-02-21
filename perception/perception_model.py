import torch
import torch.nn as nn
import torch.nn.functional as F


class MobileNetUNet(nn.Module):
    def __init__(self, n_semantic_classes):
        super(MobileNetUNet, self).__init__()
        self.encoder = MobileNetEncoder()
        self.rgb_decoder = UNetDecoder(3)
        self.semantic_seg_decoder = UNetDecoder(n_semantic_classes)
        self.depth_decoder = UNetDecoder(1)

        self.use_rgb_decoder = True

    def forward(self, x: torch.Tensor):
        # Encode input
        x = self.encoder(x)

        # Decode into RGB, semantic segmentation, and depth map.
        semantic_seg_pred = F.softmax(self.semantic_seg_decoder(x), dim=1)
        depth_pred = torch.sigmoid(self.depth_decoder(x))
        if self.use_rgb_decoder:
            rgb_pred = F.relu(self.rgb_decoder(x))
            return rgb_pred, semantic_seg_pred, depth_pred

        return semantic_seg_pred, depth_pred

    def set_rgb_decoder(self, use_rgb_decoder: bool):
        self.use_rgb_decoder = use_rgb_decoder

    def get_encoder(self):
        return self.encoder


class MobileNetEncoder(nn.Module):
    def __init__(self):
        super(MobileNetEncoder, self).__init__()
        # Input Conv layer
        self.input_conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=(1, 1))
        self.intput_batch_norm1 = nn.BatchNorm2d(32)
        self.input_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=(1, 1))
        self.intput_batch_norm2 = nn.BatchNorm2d(32)

        # For every second depthwise separable conv operation, the resolution is downsampled by using stride = (2,2)
        self.depthwise_separable1 = DepthwiseSeparableConv(32, 64)
        self.depthwise_separable2 = DepthwiseSeparableConv(64, 128, depthwise_stride=(2, 2))
        self.depthwise_separable3 = DepthwiseSeparableConv(128, 128)
        self.depthwise_separable4 = DepthwiseSeparableConv(128, 256, depthwise_stride=(2, 2))
        self.depthwise_separable5 = DepthwiseSeparableConv(256, 256)
        self.depthwise_separable6 = DepthwiseSeparableConv(256, 512, depthwise_stride=(2, 2))

        # Block of five repeated depthwise separable conv operations
        self.depthwise_separable7 = DepthwiseSeparableConv(512, 512)
        self.depthwise_separable8 = DepthwiseSeparableConv(512, 512)
        self.depthwise_separable9 = DepthwiseSeparableConv(512, 512)
        self.depthwise_separable10 = DepthwiseSeparableConv(512, 512)
        self.depthwise_separable11 = DepthwiseSeparableConv(512, 512)

        # The two final depthwise separable conv operations, outputting 1024 feature maps
        self.depthwise_separable12 = DepthwiseSeparableConv(512, 512, depthwise_stride=(2, 2))
        self.depthwise_separable13 = DepthwiseSeparableConv(512, 1024)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.intput_batch_norm1(self.input_conv1(x)))
        f1 = x
        x = F.relu(self.intput_batch_norm2(self.input_conv2(x)))
        x = self.depthwise_separable1(x)
        f2 = x
        x = self.depthwise_separable2(x)
        x = self.depthwise_separable3(x)
        f3 = x
        x = self.depthwise_separable4(x)
        x = self.depthwise_separable5(x)
        f4 = x
        x = self.depthwise_separable6(x)
        x = self.depthwise_separable7(x)
        x = self.depthwise_separable8(x)
        x = self.depthwise_separable9(x)
        x = self.depthwise_separable10(x)
        x = self.depthwise_separable11(x)
        f5 = x
        x = self.depthwise_separable12(x)
        x = self.depthwise_separable13(x)
        f6 = x
        return [f1, f2, f3, f4, f5, f6]


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, input_channels, output_channels, depthwise_kernel_size=3, depthwise_stride=(1, 1),
                 padding=(1, 1),
                 padding_mode="zeros"):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels=input_channels, out_channels=input_channels,
                                   kernel_size=depthwise_kernel_size, stride=depthwise_stride,
                                   padding=padding, padding_mode=padding_mode, groups=input_channels)
        self.batchnorm1 = nn.BatchNorm2d(input_channels)
        self.pointwise = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1)
        self.batchnorm2 = nn.BatchNorm2d(output_channels)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.batchnorm1(self.depthwise(x)))
        x = F.relu(self.batchnorm2(self.pointwise(x)))
        return x


class UNetDecoder(nn.Module):
    def __init__(self, n_classes):
        """
        UNet consists of an encoder (contracting path) and a decoder (expansive path).
        This is actually just the implementation of the decoder (i.e. the expansive path).
        """
        super(UNetDecoder, self).__init__()
        self.init_conv = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.init_batch_norm = nn.BatchNorm2d(512)
        self.up1 = UpConv(1024, 256)
        self.up2 = UpConv(512, 128)
        self.up3 = UpConv(256, 64)
        self.up4 = UpConv(128, 32)
        self.up5 = UpConv(64, 32)
        self.conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, features):
        f1, f2, f3, f4, f5, f6 = features
        x = F.relu(self.init_batch_norm(self.init_conv(f6)))
        x = self.up1(x, f5)
        x = self.up2(x, f4)
        x = self.up3(x, f3)
        x = self.up4(x, f2)
        x = self.up5(x, f1)
        x = self.conv(x)
        x = self.final_conv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UpConv, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(output_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        x1 = self.upsample(x1)
        # dim 0 is batch dimension, dim 1 is channel dimension.
        x1 = torch.cat([x2, x1], dim=1)
        x1 = F.relu(self.batch_norm1(self.conv1(x1)))
        x1 = F.relu(self.batch_norm2(self.conv2(x1)))
        return x1
