import torch
import torch.nn as nn
import torch.nn.functional as F


class ShuffleOp(nn.Module):
    def __init__(self, groups):
        super(ShuffleOp, self).__init__()
        self.groups = groups

    def forward(self, x):
        """channel shuffle: [N, C, H, W] -> [N, g, C/g, H, W] -> [N, C/g, g, H, w] -> [N, C, H, W]"""
        N, C, H, W = x.shape
        g = self.groups
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)


class HRChannelExpandSpatialDownsample(nn.Module):
    def __init__(self, in_channels, out_channels=None, downsample_rate=1, min_channels_of_a_grp=4):
        super().__init__()
        out_channels = out_channels or in_channels
        self.avg_pool = nn.AvgPool2d(kernel_size=downsample_rate, stride=downsample_rate)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels // min_channels_of_a_grp)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.relu3(self.bn3(self.conv3(x)))
        return x


class HRChannelShrinkSpatialUpsample(nn.Module):
    def __init__(
            self, in_channels, out_channels=None, upsample_rate=1, min_channels_of_a_grp=4, upsample_mode='nearest'
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels // min_channels_of_a_grp)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=upsample_rate, mode=upsample_mode)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.upsample(x)
        return x


class HRResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, min_channels_of_a_grp=4):
        super().__init__()
        out_channels = out_channels or in_channels
        # if out_channel_moment is None:
        #     out_channel_moment = "pre" if out_channels < in_channels else "post"
        # assert out_channel_moment in ["pre", "post"], f"only [pre, post] are valid, but {out_channel_moment!r} given."
        # mid_channels = in_channels if out_channel_moment == "post" else out_channels
        mid_channels = min(in_channels, out_channels)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1, groups=mid_channels // min_channels_of_a_grp)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.relu3(self.bn3(self.conv3(x)))
        return x


class HRSplitConcatShuffleBlock(nn.Module):
    def __init__(self, in_channels, min_channels_of_a_grp=4):
        super().__init__()
        half_channels = in_channels // 2
        self.conv1 = nn.Conv2d(half_channels, half_channels, 1)
        self.bn1 = nn.BatchNorm2d(half_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(half_channels, half_channels, 3, padding=1, groups=half_channels // min_channels_of_a_grp)
        self.bn2 = nn.BatchNorm2d(half_channels)
        self.conv3 = nn.Conv2d(half_channels, half_channels, 1)
        self.bn3 = nn.BatchNorm2d(half_channels)
        self.relu3 = nn.ReLU()
        self.shuffle_op = ShuffleOp(2)

    def forward(self, x):
        c = x.shape[1]
        assert c % 2 == 0, f"channels must be a even number, but {c!r} is found."
        x1, x2 = x[:, :c // 2, ...], x[:, c // 2:, ...]
        x2 = self.relu1(self.bn1(self.conv1(x2)))
        x2 = self.bn2(self.conv2(x2))
        x2 = self.relu3(self.bn3(self.conv3(x2)))
        x = torch.cat((x1, x2), dim=1)
        x = self.shuffle_op(x)
        return x


class HighResolutionNet(nn.Module):
    def __init__(self, layer_settings=None, upsample_mode='nearest'):
        super().__init__()
        if layer_settings is None:
            layer_settings = (
                ("r4", 24, 4),
                ("r8", 48, 4),
                ("r16", 96, 4),
                ("r32", 192, 4),
            )
        c4, c8, c16, c32 = [s[1] for s in layer_settings]

        ################
        #    Stage 0
        ################
        self.first_conv = nn.Conv2d(3, c4, 3, stride=2, padding=1)
        self.first_bn = nn.BatchNorm2d(c4)
        self.second_conv = nn.Conv2d(c4, c4, 3, stride=2, padding=1)
        self.second_bn = nn.BatchNorm2d(c4)
        self.second_relu = nn.ReLU()

        ################
        #    Stage 1
        ################
        self.branch_s1_r4a = HRSplitConcatShuffleBlock(c4)
        self.branch_s1_r4b = HRSplitConcatShuffleBlock(c4)
        self.branch_s1_r4c = HRSplitConcatShuffleBlock(c4)
        self.branch_s1_r4d = HRSplitConcatShuffleBlock(c4)
        self.s1_r4r4 = HRSplitConcatShuffleBlock(c4)
        self.s1_r4r8 = HRChannelExpandSpatialDownsample(c4, out_channels=c8, downsample_rate=2)

        ################
        #    Stage 2
        ################
        self.branch_s2_r4a = HRSplitConcatShuffleBlock(c4)
        self.branch_s2_r4b = HRSplitConcatShuffleBlock(c4)
        self.branch_s2_r4c = HRSplitConcatShuffleBlock(c4)
        self.branch_s2_r4d = HRSplitConcatShuffleBlock(c4)
        self.branch_s2_r8a = HRSplitConcatShuffleBlock(c8)
        self.branch_s2_r8b = HRSplitConcatShuffleBlock(c8)
        self.branch_s2_r8c = HRSplitConcatShuffleBlock(c8)
        self.branch_s2_r8d = HRSplitConcatShuffleBlock(c8)
        self.s2_r4r4 = HRSplitConcatShuffleBlock(c4)
        self.s2_r4r8 = HRChannelExpandSpatialDownsample(c4, out_channels=c8, downsample_rate=2)
        self.s2_r4r16 = HRChannelExpandSpatialDownsample(c4, out_channels=c16, downsample_rate=4)
        self.s2_r8r8 = HRSplitConcatShuffleBlock(c8)
        self.s2_r8r4 = HRChannelShrinkSpatialUpsample(c8, out_channels=c4, upsample_rate=2, upsample_mode=upsample_mode)
        self.s2_r8r16 = HRChannelExpandSpatialDownsample(c8, out_channels=c16, downsample_rate=2)

        ################
        #    Stage 3
        ################
        self.branch_s3_r4a = HRSplitConcatShuffleBlock(c4)
        self.branch_s3_r4b = HRSplitConcatShuffleBlock(c4)
        self.branch_s3_r4c = HRSplitConcatShuffleBlock(c4)
        self.branch_s3_r4d = HRSplitConcatShuffleBlock(c4)
        self.branch_s3_r8a = HRSplitConcatShuffleBlock(c8)
        self.branch_s3_r8b = HRSplitConcatShuffleBlock(c8)
        self.branch_s3_r8c = HRSplitConcatShuffleBlock(c8)
        self.branch_s3_r8d = HRSplitConcatShuffleBlock(c8)
        self.branch_s3_r16a = HRSplitConcatShuffleBlock(c16)
        self.branch_s3_r16b = HRSplitConcatShuffleBlock(c16)
        self.branch_s3_r16c = HRSplitConcatShuffleBlock(c16)
        self.branch_s3_r16d = HRSplitConcatShuffleBlock(c16)
        self.s3_r4r4 = HRSplitConcatShuffleBlock(c4)
        self.s3_r4r8 = HRChannelExpandSpatialDownsample(c4, out_channels=c8, downsample_rate=2)
        self.s3_r4r16 = HRChannelExpandSpatialDownsample(c4, out_channels=c16, downsample_rate=4)
        self.s3_r4r32 = HRChannelExpandSpatialDownsample(c4, out_channels=c32, downsample_rate=8)
        self.s3_r8r8 = HRSplitConcatShuffleBlock(c8)
        self.s3_r8r4 = HRChannelShrinkSpatialUpsample(c8, out_channels=c4, upsample_rate=2, upsample_mode=upsample_mode)
        self.s3_r8r16 = HRChannelExpandSpatialDownsample(c8, out_channels=c16, downsample_rate=2)
        self.s3_r8r32 = HRChannelExpandSpatialDownsample(c8, out_channels=c32, downsample_rate=4)
        self.s3_r16r16 = HRSplitConcatShuffleBlock(c16)
        self.s3_r16r4 = HRChannelShrinkSpatialUpsample(c16, out_channels=c4, upsample_rate=4, upsample_mode=upsample_mode)
        self.s3_r16r8 = HRChannelShrinkSpatialUpsample(c16, out_channels=c8, upsample_rate=2, upsample_mode=upsample_mode)
        self.s3_r16r32 = HRChannelExpandSpatialDownsample(c16, out_channels=c32, downsample_rate=2)

        ################
        #    Stage 4
        ################
        self.branch_s4_r4a = HRSplitConcatShuffleBlock(c4)
        self.branch_s4_r4b = HRSplitConcatShuffleBlock(c4)
        self.branch_s4_r4c = HRSplitConcatShuffleBlock(c4)
        self.branch_s4_r4d = HRSplitConcatShuffleBlock(c4)
        self.branch_s4_r8a = HRSplitConcatShuffleBlock(c8)
        self.branch_s4_r8b = HRSplitConcatShuffleBlock(c8)
        self.branch_s4_r8c = HRSplitConcatShuffleBlock(c8)
        self.branch_s4_r8d = HRSplitConcatShuffleBlock(c8)
        self.branch_s4_r16a = HRSplitConcatShuffleBlock(c16)
        self.branch_s4_r16b = HRSplitConcatShuffleBlock(c16)
        self.branch_s4_r16c = HRSplitConcatShuffleBlock(c16)
        self.branch_s4_r16d = HRSplitConcatShuffleBlock(c16)
        self.branch_s4_r32a = HRSplitConcatShuffleBlock(c32)
        self.branch_s4_r32b = HRSplitConcatShuffleBlock(c32)
        self.branch_s4_r32c = HRSplitConcatShuffleBlock(c32)
        self.branch_s4_r32d = HRSplitConcatShuffleBlock(c32)
        self.s4_r4r4 = HRSplitConcatShuffleBlock(c4)
        self.s4_r4r8 = HRChannelExpandSpatialDownsample(c4, out_channels=c8, downsample_rate=2)
        self.s4_r4r16 = HRChannelExpandSpatialDownsample(c4, out_channels=c16, downsample_rate=4)
        self.s4_r4r32 = HRChannelExpandSpatialDownsample(c4, out_channels=c32, downsample_rate=8)
        self.s4_r8r8 = HRSplitConcatShuffleBlock(c8)
        self.s4_r8r4 = HRChannelShrinkSpatialUpsample(c8, out_channels=c4, upsample_rate=2, upsample_mode=upsample_mode)
        self.s4_r8r16 = HRChannelExpandSpatialDownsample(c8, out_channels=c16, downsample_rate=2)
        self.s4_r8r32 = HRChannelExpandSpatialDownsample(c8, out_channels=c32, downsample_rate=4)
        self.s4_r16r16 = HRSplitConcatShuffleBlock(c16)
        self.s4_r16r4 = HRChannelShrinkSpatialUpsample(c16, out_channels=c4, upsample_rate=4, upsample_mode=upsample_mode)
        self.s4_r16r8 = HRChannelShrinkSpatialUpsample(c16, out_channels=c8, upsample_rate=2, upsample_mode=upsample_mode)
        self.s4_r16r32 = HRChannelExpandSpatialDownsample(c16, out_channels=c32, downsample_rate=2)
        self.s4_r32r32 = HRSplitConcatShuffleBlock(c32)
        self.s4_r32r4 = HRChannelShrinkSpatialUpsample(c32, out_channels=c4, upsample_rate=8, upsample_mode=upsample_mode)
        self.s4_r32r8 = HRChannelShrinkSpatialUpsample(c32, out_channels=c8, upsample_rate=4, upsample_mode=upsample_mode)
        self.s4_r32r16 = HRChannelShrinkSpatialUpsample(c32, out_channels=c16, upsample_rate=2, upsample_mode=upsample_mode)

        ################
        #    Final
        ################
        self.up_r8r4 = nn.Upsample(scale_factor=2, mode=upsample_mode)
        self.up_r16r4 = nn.Upsample(scale_factor=4, mode=upsample_mode)
        self.up_r32r4 = nn.Upsample(scale_factor=8, mode=upsample_mode)

        self.initialize_params()

    def forward(self, x):
        # Stage 0
        x = self.first_bn(self.first_conv(x))
        x = self.second_relu(self.second_bn(self.second_conv(x)))

        # Stage 1
        r4 = self.branch_s1_r4d(self.branch_s1_r4c(self.branch_s1_r4b(self.branch_s1_r4a(x))))
        r8 = self.s1_r4r8(r4)
        r4 = self.s1_r4r4(r4)

        # Stage 2
        r4 = self.branch_s2_r4d(self.branch_s2_r4c(self.branch_s2_r4b(self.branch_s2_r4a(r4))))
        r8 = self.branch_s2_r8d(self.branch_s2_r8c(self.branch_s2_r8b(self.branch_s2_r8a(r8))))
        r16 = self.s2_r4r16(r4) + self.s2_r8r16(r8)
        fused_r8 = self.s2_r4r8(r4) + self.s2_r8r8(r8)
        fused_r4 = self.s2_r4r4(r4) + self.s2_r8r4(r8)
        r8 = fused_r8
        r4 = fused_r4

        # Stage 3
        r4 = self.branch_s3_r4d(self.branch_s3_r4c(self.branch_s3_r4b(self.branch_s3_r4a(r4))))
        r8 = self.branch_s3_r8d(self.branch_s3_r8c(self.branch_s3_r8b(self.branch_s3_r8a(r8))))
        r16 = self.branch_s3_r16d(self.branch_s3_r16c(self.branch_s3_r16b(self.branch_s3_r16a(r16))))
        r32 = self.s3_r4r32(r4) + self.s3_r8r32(r8) + self.s3_r16r32(r16)
        fused_r16 = self.s3_r4r16(r4) + self.s3_r8r16(r8) + self.s3_r16r16(r16)
        fused_r8 = self.s3_r4r8(r4) + self.s3_r8r8(r8) + self.s3_r16r8(r16)
        fused_r4 = self.s3_r4r4(r4) + self.s3_r8r4(r8) + self.s3_r16r4(r16)
        r16 = fused_r16
        r8 = fused_r8
        r4 = fused_r4

        # Stage 4
        r4 = self.branch_s4_r4d(self.branch_s4_r4c(self.branch_s4_r4b(self.branch_s4_r4a(r4))))
        r8 = self.branch_s4_r8d(self.branch_s4_r8c(self.branch_s4_r8b(self.branch_s4_r8a(r8))))
        r16 = self.branch_s4_r16d(self.branch_s4_r16c(self.branch_s4_r16b(self.branch_s4_r16a(r16))))
        r32 = self.branch_s4_r32d(self.branch_s4_r32c(self.branch_s4_r32b(self.branch_s4_r32a(r32))))
        fused_r32 = self.s4_r4r32(r4) + self.s4_r8r32(r8) + self.s4_r16r32(r16) + self.s4_r32r32(r32)
        fused_r16 = self.s4_r4r16(r4) + self.s4_r8r16(r8) + self.s4_r16r16(r16) + self.s4_r32r16(r32)
        fused_r8 = self.s4_r4r8(r4) + self.s4_r8r8(r8) + self.s4_r16r8(r16) + self.s4_r32r8(r32)
        fused_r4 = self.s4_r4r4(r4) + self.s4_r8r4(r8) + self.s4_r16r4(r16) + self.s4_r32r4(r32)

        # Final
        r8_up = self.up_r8r4(fused_r8)
        r16_up = self.up_r16r4(fused_r16)
        r32_up = self.up_r32r4(fused_r32)

        return torch.cat((fused_r4, r8_up, r16_up, r32_up), dim=1)

    def initialize_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.data.zero_()
            # !!!IMPORTANT: batchnorm must be initialized and initialized as follows to meet quantization requirements.
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.running_var, 1)
                nn.init.constant_(m.running_mean, 0)


class HigherResolutionNet(nn.Module):
    def __init__(self, layer_settings=None, upsample_mode='nearest'):
        super().__init__()
        if layer_settings is None:
            layer_settings = (
                ("r2", 12, 4),
                ("r4", 24, 4),
                ("r8", 48, 4),
                ("r16", 96, 4),
                ("r32", 192, 4),
            )
        c2, c4, c8, c16, c32 = [s[1] for s in layer_settings]

        ################
        #    Stage 0
        ################
        self.first_conv = nn.Conv2d(3, c4, 3, stride=2, padding=1)
        self.first_bn = nn.BatchNorm2d(c4)
        self.first_relu = nn.ReLU()
        self.branch_s0_r2 = HRResBlock(c4, out_channels=c2)
        self.branch_s0_r4 = HRChannelExpandSpatialDownsample(c4, out_channels=c4, downsample_rate=2)

        ################
        #    Stage 1
        ################
        self.branch_s1_r2a = HRResBlock(c2)
        self.branch_s1_r2b = HRResBlock(c2)
        self.branch_s1_r2c = HRResBlock(c2)
        self.branch_s1_r2d = HRResBlock(c2)
        self.branch_s1_r4a = HRSplitConcatShuffleBlock(c4)
        self.branch_s1_r4b = HRSplitConcatShuffleBlock(c4)
        self.branch_s1_r4c = HRSplitConcatShuffleBlock(c4)
        self.branch_s1_r4d = HRSplitConcatShuffleBlock(c4)
        self.s1_r2r2 = HRResBlock(c2)
        self.s1_r2r4 = HRChannelExpandSpatialDownsample(c2, out_channels=c4, downsample_rate=2)
        self.s1_r2r8 = HRChannelExpandSpatialDownsample(c2, out_channels=c8, downsample_rate=4)
        self.s1_r4r2 = HRChannelShrinkSpatialUpsample(c4, out_channels=c2, upsample_rate=2, upsample_mode=upsample_mode)
        self.s1_r4r4 = HRSplitConcatShuffleBlock(c4)
        self.s1_r4r8 = HRChannelExpandSpatialDownsample(c4, out_channels=c8, downsample_rate=2)

        ################
        #    Stage 2
        ################
        self.branch_s2_r2a = HRResBlock(c2)
        self.branch_s2_r2b = HRResBlock(c2)
        self.branch_s2_r2c = HRResBlock(c2)
        self.branch_s2_r2d = HRResBlock(c2)
        self.branch_s2_r4a = HRSplitConcatShuffleBlock(c4)
        self.branch_s2_r4b = HRSplitConcatShuffleBlock(c4)
        self.branch_s2_r4c = HRSplitConcatShuffleBlock(c4)
        self.branch_s2_r4d = HRSplitConcatShuffleBlock(c4)
        self.branch_s2_r8a = HRSplitConcatShuffleBlock(c8)
        self.branch_s2_r8b = HRSplitConcatShuffleBlock(c8)
        self.branch_s2_r8c = HRSplitConcatShuffleBlock(c8)
        self.branch_s2_r8d = HRSplitConcatShuffleBlock(c8)
        self.s2_r2r2 = HRResBlock(c2)
        self.s2_r2r4 = HRChannelExpandSpatialDownsample(c2, out_channels=c4, downsample_rate=2)
        self.s2_r2r8 = HRChannelExpandSpatialDownsample(c2, out_channels=c8, downsample_rate=4)
        self.s2_r2r16 = HRChannelExpandSpatialDownsample(c2, out_channels=c16, downsample_rate=8)
        self.s2_r4r2 = HRChannelShrinkSpatialUpsample(c4, out_channels=c2, upsample_rate=2, upsample_mode=upsample_mode)
        self.s2_r4r4 = HRSplitConcatShuffleBlock(c4)
        self.s2_r4r8 = HRChannelExpandSpatialDownsample(c4, out_channels=c8, downsample_rate=2)
        self.s2_r4r16 = HRChannelExpandSpatialDownsample(c4, out_channels=c16, downsample_rate=4)
        self.s2_r8r2 = HRChannelShrinkSpatialUpsample(c8, out_channels=c2, upsample_rate=4, upsample_mode=upsample_mode)
        self.s2_r8r4 = HRChannelShrinkSpatialUpsample(c8, out_channels=c4, upsample_rate=2, upsample_mode=upsample_mode)
        self.s2_r8r8 = HRSplitConcatShuffleBlock(c8)
        self.s2_r8r16 = HRChannelExpandSpatialDownsample(c8, out_channels=c16, downsample_rate=2)

        ################
        #    Stage 3
        ################
        self.branch_s3_r2a = HRResBlock(c2)
        self.branch_s3_r2b = HRResBlock(c2)
        self.branch_s3_r2c = HRResBlock(c2)
        self.branch_s3_r2d = HRResBlock(c2)
        self.branch_s3_r4a = HRSplitConcatShuffleBlock(c4)
        self.branch_s3_r4b = HRSplitConcatShuffleBlock(c4)
        self.branch_s3_r4c = HRSplitConcatShuffleBlock(c4)
        self.branch_s3_r4d = HRSplitConcatShuffleBlock(c4)
        self.branch_s3_r8a = HRSplitConcatShuffleBlock(c8)
        self.branch_s3_r8b = HRSplitConcatShuffleBlock(c8)
        self.branch_s3_r8c = HRSplitConcatShuffleBlock(c8)
        self.branch_s3_r8d = HRSplitConcatShuffleBlock(c8)
        self.branch_s3_r16a = HRSplitConcatShuffleBlock(c16)
        self.branch_s3_r16b = HRSplitConcatShuffleBlock(c16)
        self.branch_s3_r16c = HRSplitConcatShuffleBlock(c16)
        self.branch_s3_r16d = HRSplitConcatShuffleBlock(c16)
        self.s3_r2r2 = HRResBlock(c2)
        self.s3_r2r4 = HRChannelExpandSpatialDownsample(c2, out_channels=c4, downsample_rate=2)
        self.s3_r2r8 = HRChannelExpandSpatialDownsample(c2, out_channels=c8, downsample_rate=4)
        self.s3_r2r16 = HRChannelExpandSpatialDownsample(c2, out_channels=c16, downsample_rate=8)
        self.s3_r2r32 = HRChannelExpandSpatialDownsample(c2, out_channels=c32, downsample_rate=16)
        self.s3_r4r2 = HRChannelShrinkSpatialUpsample(c4, out_channels=c2, upsample_rate=2, upsample_mode=upsample_mode)
        self.s3_r4r4 = HRSplitConcatShuffleBlock(c4)
        self.s3_r4r8 = HRChannelExpandSpatialDownsample(c4, out_channels=c8, downsample_rate=2)
        self.s3_r4r16 = HRChannelExpandSpatialDownsample(c4, out_channels=c16, downsample_rate=4)
        self.s3_r4r32 = HRChannelExpandSpatialDownsample(c4, out_channels=c32, downsample_rate=8)
        self.s3_r8r2 = HRChannelShrinkSpatialUpsample(c8, out_channels=c2, upsample_rate=4, upsample_mode=upsample_mode)
        self.s3_r8r4 = HRChannelShrinkSpatialUpsample(c8, out_channels=c4, upsample_rate=2, upsample_mode=upsample_mode)
        self.s3_r8r8 = HRSplitConcatShuffleBlock(c8)
        self.s3_r8r16 = HRChannelExpandSpatialDownsample(c8, out_channels=c16, downsample_rate=2)
        self.s3_r8r32 = HRChannelExpandSpatialDownsample(c8, out_channels=c32, downsample_rate=4)
        self.s3_r16r2 = HRChannelShrinkSpatialUpsample(c16, out_channels=c2, upsample_rate=8, upsample_mode=upsample_mode)
        self.s3_r16r4 = HRChannelShrinkSpatialUpsample(c16, out_channels=c4, upsample_rate=4, upsample_mode=upsample_mode)
        self.s3_r16r8 = HRChannelShrinkSpatialUpsample(c16, out_channels=c8, upsample_rate=2, upsample_mode=upsample_mode)
        self.s3_r16r16 = HRSplitConcatShuffleBlock(c16)
        self.s3_r16r32 = HRChannelExpandSpatialDownsample(c16, out_channels=c32, downsample_rate=2)

        ################
        #    Stage 4
        ################
        self.branch_s4_r2a = HRResBlock(c2)
        self.branch_s4_r2b = HRResBlock(c2)
        self.branch_s4_r2c = HRResBlock(c2)
        self.branch_s4_r2d = HRResBlock(c2)
        self.branch_s4_r4a = HRSplitConcatShuffleBlock(c4)
        self.branch_s4_r4b = HRSplitConcatShuffleBlock(c4)
        self.branch_s4_r4c = HRSplitConcatShuffleBlock(c4)
        self.branch_s4_r4d = HRSplitConcatShuffleBlock(c4)
        self.branch_s4_r8a = HRSplitConcatShuffleBlock(c8)
        self.branch_s4_r8b = HRSplitConcatShuffleBlock(c8)
        self.branch_s4_r8c = HRSplitConcatShuffleBlock(c8)
        self.branch_s4_r8d = HRSplitConcatShuffleBlock(c8)
        self.branch_s4_r16a = HRSplitConcatShuffleBlock(c16)
        self.branch_s4_r16b = HRSplitConcatShuffleBlock(c16)
        self.branch_s4_r16c = HRSplitConcatShuffleBlock(c16)
        self.branch_s4_r16d = HRSplitConcatShuffleBlock(c16)
        self.branch_s4_r32a = HRSplitConcatShuffleBlock(c32)
        self.branch_s4_r32b = HRSplitConcatShuffleBlock(c32)
        self.branch_s4_r32c = HRSplitConcatShuffleBlock(c32)
        self.branch_s4_r32d = HRSplitConcatShuffleBlock(c32)
        self.s4_r2r2 = HRResBlock(c2)
        self.s4_r2r4 = HRChannelExpandSpatialDownsample(c2, out_channels=c4, downsample_rate=2)
        self.s4_r2r8 = HRChannelExpandSpatialDownsample(c2, out_channels=c8, downsample_rate=4)
        self.s4_r2r16 = HRChannelExpandSpatialDownsample(c2, out_channels=c16, downsample_rate=8)
        self.s4_r2r32 = HRChannelExpandSpatialDownsample(c2, out_channels=c32, downsample_rate=16)
        self.s4_r4r2 = HRChannelShrinkSpatialUpsample(c4, out_channels=c2, upsample_rate=2, upsample_mode=upsample_mode)
        self.s4_r4r4 = HRSplitConcatShuffleBlock(c4)
        self.s4_r4r8 = HRChannelExpandSpatialDownsample(c4, out_channels=c8, downsample_rate=2)
        self.s4_r4r16 = HRChannelExpandSpatialDownsample(c4, out_channels=c16, downsample_rate=4)
        self.s4_r4r32 = HRChannelExpandSpatialDownsample(c4, out_channels=c32, downsample_rate=8)
        self.s4_r8r2 = HRChannelShrinkSpatialUpsample(c8, out_channels=c2, upsample_rate=4, upsample_mode=upsample_mode)
        self.s4_r8r4 = HRChannelShrinkSpatialUpsample(c8, out_channels=c4, upsample_rate=2, upsample_mode=upsample_mode)
        self.s4_r8r8 = HRSplitConcatShuffleBlock(c8)
        self.s4_r8r16 = HRChannelExpandSpatialDownsample(c8, out_channels=c16, downsample_rate=2)
        self.s4_r8r32 = HRChannelExpandSpatialDownsample(c8, out_channels=c32, downsample_rate=4)
        self.s4_r16r2 = HRChannelShrinkSpatialUpsample(c16, out_channels=c2, upsample_rate=8, upsample_mode=upsample_mode)
        self.s4_r16r4 = HRChannelShrinkSpatialUpsample(c16, out_channels=c4, upsample_rate=4, upsample_mode=upsample_mode)
        self.s4_r16r8 = HRChannelShrinkSpatialUpsample(c16, out_channels=c8, upsample_rate=2, upsample_mode=upsample_mode)
        self.s4_r16r16 = HRSplitConcatShuffleBlock(c16)
        self.s4_r16r32 = HRChannelExpandSpatialDownsample(c16, out_channels=c32, downsample_rate=2)
        self.s4_r32r2 = HRChannelShrinkSpatialUpsample(c32, out_channels=c2, upsample_rate=16, upsample_mode=upsample_mode)
        self.s4_r32r4 = HRChannelShrinkSpatialUpsample(c32, out_channels=c4, upsample_rate=8, upsample_mode=upsample_mode)
        self.s4_r32r8 = HRChannelShrinkSpatialUpsample(c32, out_channels=c8, upsample_rate=4, upsample_mode=upsample_mode)
        self.s4_r32r16 = HRChannelShrinkSpatialUpsample(c32, out_channels=c16, upsample_rate=2, upsample_mode=upsample_mode)
        self.s4_r32r32 = HRSplitConcatShuffleBlock(c32)

        ################
        #    Final
        ################
        self.up_r4r2 = nn.Upsample(scale_factor=2, mode=upsample_mode)
        self.up_r8r2 = nn.Upsample(scale_factor=4, mode=upsample_mode)
        self.up_r16r2 = nn.Upsample(scale_factor=8, mode=upsample_mode)
        self.up_r32r2 = nn.Upsample(scale_factor=16, mode=upsample_mode)

        self.initialize_params()

    def forward(self, x):
        # Stage 0
        x = self.first_relu(self.first_bn(self.first_conv(x)))
        r2 = self.branch_s0_r2(x)
        r4 = self.branch_s0_r4(x)

        # Stage 1
        r2 = self.branch_s1_r2d(self.branch_s1_r2c(self.branch_s1_r2b(self.branch_s1_r2a(r2))))
        r4 = self.branch_s1_r4d(self.branch_s1_r4c(self.branch_s1_r4b(self.branch_s1_r4a(r4))))
        r8 = self.s1_r2r8(r2) + self.s1_r4r8(r4)
        fused_r4 = self.s1_r2r4(r2) + self.s1_r4r4(r4)
        fused_r2 = self.s1_r2r2(r2) + self.s1_r4r2(r4)
        r4 = fused_r4
        r2 = fused_r2

        # Stage 2
        r2 = self.branch_s2_r2d(self.branch_s2_r2c(self.branch_s2_r2b(self.branch_s2_r2a(r2))))
        r4 = self.branch_s2_r4d(self.branch_s2_r4c(self.branch_s2_r4b(self.branch_s2_r4a(r4))))
        r8 = self.branch_s2_r8d(self.branch_s2_r8c(self.branch_s2_r8b(self.branch_s2_r8a(r8))))
        r16 = self.s2_r2r16(r2) + self.s2_r4r16(r4) + self.s2_r8r16(r8)
        fused_r8 = self.s2_r2r8(r2) + self.s2_r4r8(r4) + self.s2_r8r8(r8)
        fused_r4 = self.s2_r2r4(r2) + self.s2_r4r4(r4) + self.s2_r8r4(r8)
        fused_r2 = self.s2_r2r2(r2) + self.s2_r4r2(r4) + self.s2_r8r2(r8)
        r8 = fused_r8
        r4 = fused_r4
        r2 = fused_r2

        # Stage 3
        r2 = self.branch_s3_r2d(self.branch_s3_r2c(self.branch_s3_r2b(self.branch_s3_r2a(r2))))
        r4 = self.branch_s3_r4d(self.branch_s3_r4c(self.branch_s3_r4b(self.branch_s3_r4a(r4))))
        r8 = self.branch_s3_r8d(self.branch_s3_r8c(self.branch_s3_r8b(self.branch_s3_r8a(r8))))
        r16 = self.branch_s3_r16d(self.branch_s3_r16c(self.branch_s3_r16b(self.branch_s3_r16a(r16))))
        r32 = self.s3_r4r32(r4) + self.s3_r8r32(r8) + self.s3_r16r32(r16)
        fused_r16 = self.s3_r2r16(r2) + self.s3_r4r16(r4) + self.s3_r8r16(r8) + self.s3_r16r16(r16)
        fused_r8 = self.s3_r2r8(r2) + self.s3_r4r8(r4) + self.s3_r8r8(r8) + self.s3_r16r8(r16)
        fused_r4 = self.s3_r2r4(r2) + self.s3_r4r4(r4) + self.s3_r8r4(r8) + self.s3_r16r4(r16)
        fused_r2 = self.s3_r2r2(r2) + self.s3_r4r2(r4) + self.s3_r8r2(r8) + self.s3_r16r2(r16)
        r16 = fused_r16
        r8 = fused_r8
        r4 = fused_r4
        r2 = fused_r2

        # Stage 4
        r2 = self.branch_s4_r2d(self.branch_s4_r2c(self.branch_s4_r2b(self.branch_s4_r2a(r2))))
        r4 = self.branch_s4_r4d(self.branch_s4_r4c(self.branch_s4_r4b(self.branch_s4_r4a(r4))))
        r8 = self.branch_s4_r8d(self.branch_s4_r8c(self.branch_s4_r8b(self.branch_s4_r8a(r8))))
        r16 = self.branch_s4_r16d(self.branch_s4_r16c(self.branch_s4_r16b(self.branch_s4_r16a(r16))))
        r32 = self.branch_s4_r32d(self.branch_s4_r32c(self.branch_s4_r32b(self.branch_s4_r32a(r32))))
        fused_r32 = self.s4_r2r32(r2) + self.s4_r4r32(r4) + self.s4_r8r32(r8) + self.s4_r16r32(r16) + self.s4_r32r32(r32)
        fused_r16 = self.s4_r2r16(r2) + self.s4_r4r16(r4) + self.s4_r8r16(r8) + self.s4_r16r16(r16) + self.s4_r32r16(r32)
        fused_r8 = self.s4_r2r8(r2) + self.s4_r4r8(r4) + self.s4_r8r8(r8) + self.s4_r16r8(r16) + self.s4_r32r8(r32)
        fused_r4 = self.s4_r2r4(r2) + self.s4_r4r4(r4) + self.s4_r8r4(r8) + self.s4_r16r4(r16) + self.s4_r32r4(r32)
        fused_r2 = self.s4_r2r2(r2) + self.s4_r4r2(r4) + self.s4_r8r2(r8) + self.s4_r16r2(r16) + self.s4_r32r2(r32)

        # Final
        r4_up = self.up_r4r2(fused_r4)
        r8_up = self.up_r8r2(fused_r8)
        r16_up = self.up_r16r2(fused_r16)
        r32_up = self.up_r32r2(fused_r32)

        return torch.cat((fused_r2, r4_up, r8_up, r16_up, r32_up), dim=1)

    def initialize_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.data.zero_()
            # !!!IMPORTANT: batchnorm must be initialized and initialized as follows to meet quantization requirements.
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.running_var, 1)
                nn.init.constant_(m.running_mean, 0)
