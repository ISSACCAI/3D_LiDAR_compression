import torch
import torchvision
from torch.autograd import Variable
from torch.nn import functional as F
import math
from torch.optim import lr_scheduler
import sepconv
import sys
import pytorch_ssim


# 相当于转为tensor的意思
def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


# 获取vgg19的relu4_4层（35）
# modelvgg19 = torchvision.models.vgg19(pretrained=True).features[35]


class ChannelAttention(torch.nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)

        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(torch.nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = torch.nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class KernelEstimation(torch.nn.Module):
    def __init__(self, kernel_size_Vertical, kernel_size_Horizontal):
        super(KernelEstimation, self).__init__()
        self.kernel_size_Vertical = kernel_size_Vertical
        self.kernel_size_Horizontal = kernel_size_Horizontal

        def Basic(input_channel, output_channel):
            # 一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=3, stride=1,
                                padding=1),
                torch.nn.LeakyReLU(inplace=False),
                torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1,
                                padding=1),
                torch.nn.LeakyReLU(inplace=False),
                torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1,
                                padding=1),
                torch.nn.LeakyReLU(inplace=False)  # 表示输入保持不变

            )

        def Upsample(channel):
            return torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # scale_factor表示输出相比输入的倍数
                torch.nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False),

            )

        # 最后拓展为四个子网络，分别估算四个核
        def Subnet_Vertical(ks):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1),
            )
        def Subnet_Horizontal(ks):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False),
                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False),
                torch.nn.Conv2d(in_channels=128, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1),
            )


        self.moduleConv1 = Basic(6, 64)
        self.ca1 = ChannelAttention(64)
        self.sa = SpatialAttention()
        self.modulePool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        # self.moduleConv2 = Basic(32, 64)
        # self.ca2 = ChannelAttention(64)
        # self.modulePool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = Basic(64, 128)
        self.ca3 = ChannelAttention(128)
        self.modulePool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic(128, 256)
        self.ca4 = ChannelAttention(256)
        self.modulePool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv5 = Basic(256, 512)
        self.ca5 = ChannelAttention(512)
        self.modulePool5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleDeconv5 = Basic(512, 512)
        self.ca55 = ChannelAttention(512)
        self.moduleUpsample5 = Upsample(512)

        self.moduleDeconv4 = Basic(512, 256)
        self.ca44 = ChannelAttention(256)
        self.moduleUpsample4 = Upsample(256)

        self.moduleDeconv3 = Basic(256, 128)
        self.ca33 = ChannelAttention(128)
        self.moduleUpsample3 = Upsample(128)

        # self.moduleDeconv2 = Basic(128, 64)
        # self.moduleUpsample2 = Upsample(64)

        self.moduleVertical1 = Subnet_Vertical(self.kernel_size_Vertical)
        self.moduleVertical2 = Subnet_Vertical(self.kernel_size_Vertical)
        self.moduleHorizontal1 = Subnet_Horizontal(self.kernel_size_Horizontal)
        self.moduleHorizontal2 = Subnet_Horizontal(self.kernel_size_Horizontal)

    def forward(self, rfield0, rfield2):
        tensorJoin = torch.cat([rfield0, rfield2], 1)  # 按深度维拼接

        tensorConv1 = self.moduleConv1(tensorJoin)
        tensorCBAM_ca1 = self.ca1(tensorConv1) * tensorConv1
        tensorCBAM_sa1 = self.sa(tensorCBAM_ca1) * tensorCBAM_ca1
        # tensorPool1 = F.interpolate(tensorCBAM_sa1, scale_factor=0.5)
        tensorPool1 = self.modulePool1(tensorCBAM_sa1)

        # tensorConv2 = self.moduleConv2(tensorPool1)
        # tensorCBAM_ca2 = self.ca2(tensorConv2) * tensorConv2
        # tensorCBAM_sa2 = self.sa(tensorCBAM_ca2) * tensorCBAM_ca2
        # tensorPool2 = F.interpolate(tensorCBAM_sa2, scale_factor=0.5)
        # tensorPool2 = self.modulePool1(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool1)
        tensorCBAM_ca3 = self.ca3(tensorConv3) * tensorConv3
        tensorCBAM_sa3 = self.sa(tensorCBAM_ca3) * tensorCBAM_ca3
        # tensorPool3 = F.interpolate(tensorCBAM_sa3, scale_factor=0.5)
        tensorPool3 = self.modulePool1(tensorCBAM_sa3)

        tensorConv4 = self.moduleConv4(tensorPool3)
        tensorCBAM_ca4 = self.ca4(tensorConv4) * tensorConv4
        tensorCBAM_sa4 = self.sa(tensorCBAM_ca4) * tensorCBAM_ca4
        # tensorPool4 = F.interpolate(tensorCBAM_sa4, scale_factor=0.5)
        tensorPool4 = self.modulePool1(tensorCBAM_sa4)

        tensorConv5 = self.moduleConv5(tensorPool4)
        tensorCBAM_ca5 = self.ca5(tensorConv5) * tensorConv5
        tensorCBAM_sa5 = self.sa(tensorCBAM_ca5) * tensorCBAM_ca5
        # tensorPool5 = F.interpolate(tensorCBAM_sa5, scale_factor=0.5)
        tensorPool5 = self.modulePool1(tensorCBAM_sa5)

        tensorDeconv5 = self.moduleDeconv5(tensorPool5)
        tensorCBAM_ca55 = self.ca55(tensorDeconv5) * tensorDeconv5
        tensorCBAM_sa55 = self.sa(tensorCBAM_ca55) * tensorCBAM_ca55
        tensorUpsample5 = self.moduleUpsample5(tensorCBAM_sa55)

        tensorCombine = tensorUpsample5 + tensorCBAM_sa5

        tensorDeconv4 = self.moduleDeconv4(tensorCombine)
        tensorCBAM_ca44 = self.ca44(tensorDeconv4) * tensorDeconv4
        tensorCBAM_sa44 = self.sa(tensorCBAM_ca44) * tensorCBAM_ca44
        tensorUpsample4 = self.moduleUpsample4(tensorCBAM_sa44)

        tensorCombine = tensorUpsample4 + tensorCBAM_sa4

        tensorDeconv3 = self.moduleDeconv3(tensorCombine)
        tensorCBAM_ca33 = self.ca33(tensorDeconv3) * tensorDeconv3
        tensorCBAM_sa33 = self.sa(tensorCBAM_ca33) * tensorCBAM_ca33
        tensorUpsample3 = self.moduleUpsample3(tensorCBAM_sa33)

        tensorCombine = tensorUpsample3 + tensorCBAM_sa3

        # tensorDeconv2 = self.moduleDeconv2(tensorCombine)
        # tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)
        #
        # tensorCombine = tensorUpsample2 + tensorConv2

        Vertical1 = self.moduleVertical1(tensorCombine)
        Vertical2 = self.moduleVertical2(tensorCombine)
        Horizontal1 = self.moduleHorizontal1(tensorCombine)
        Horizontal2 = self.moduleHorizontal2(tensorCombine)

        return Vertical1, Horizontal1, Vertical2, Horizontal2


class SepConvNet(torch.nn.Module):
    def __init__(self, kernel_size_Vertical, kernel_size_Horizontal):
        super(SepConvNet, self).__init__()
        self.kernel_size_Vertical = kernel_size_Vertical
        self.kernel_size_Horizontal = kernel_size_Horizontal
        self.kernel_pad_Vertical = int(math.floor(kernel_size_Vertical / 2.0))  # 选取kernel的一半刚好可以计算最边缘的像素
        self.kernel_pad_Horizontal = int(math.floor(kernel_size_Horizontal / 2.0))

        self.epoch = Variable(torch.tensor(0, requires_grad=False))  # 表示self.epoch初始值为一个不可以自动求梯度的值为0的tensor
        self.get_kernel = KernelEstimation(self.kernel_size_Vertical, self.kernel_size_Horizontal)
        self.optimizer = torch.optim.Adamax(self.parameters())
        #self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=4)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        self.criterion = torch.nn.MSELoss()
        self.modulePad = torch.nn.ReplicationPad2d([self.kernel_pad_Horizontal, self.kernel_pad_Horizontal, self.kernel_pad_Vertical,
                                                    self.kernel_pad_Vertical])  # 重复填充，左右上下填充数量为self.kernel_pad，数值为边缘像素值

    def forward(self, frame0, frame2):  # 一个RGB图像转为tensor表示为[__，深度，高度/行，宽度/列]
        # 对输入两帧图像做联合归一化
        # mixframe = torch.cat([frame0, frame2], dim=2)
        # frame0 = (frame0-torch.mean(mixframe))/torch.std(mixframe)
        # frame2 = (frame2 - torch.mean(mixframe)) / (torch.std(mixframe)+0.0000001)

        h0 = int(list(frame0.size())[2])
        w0 = int(list(frame0.size())[3])
        h2 = int(list(frame2.size())[2])
        w2 = int(list(frame2.size())[3])
        if h0 != h2 or w0 != w2:
            sys.exit('Frame sizes do not match')

        h_padded = False
        w_padded = False
        if h0 % 16 != 0:  # 这里的32表示的是U型网络，编码器和解码器的层数为5，2^5=32, 防止出现在池化和上采样时候出现不能整除的截断误差
            pad_h = 16 - (h0 % 16)
            frame0 = F.pad(frame0, (0, 0, 0, pad_h))  # 在下方填充，填充默认值0
            frame2 = F.pad(frame2, (0, 0, 0, pad_h))
            h_padded = True

        if w0 % 16 != 0:
            pad_w = 16 - (w0 % 16)
            frame0 = F.pad(frame0, (0, pad_w, 0, 0))  # 在右边填充，填充默认值0
            frame2 = F.pad(frame2, (0, pad_w, 0, 0))
            w_padded = True

        Vertical1, Horizontal1, Vertical2, Horizontal2 = self.get_kernel(frame0, frame2)

        tensorDot1 = sepconv.sepconv_func.apply(self.modulePad(frame0), Vertical1, Horizontal1)
        tensorDot2 = sepconv.sepconv_func.apply(self.modulePad(frame2), Vertical2, Horizontal2)

        frame01 = torch.ones(frame0.shape[0], frame0.shape[1], frame0.shape[2], frame0.shape[3])
        frame21 = torch.ones(frame2.shape[0], frame2.shape[1], frame2.shape[2], frame2.shape[3])
        frame01 = to_variable(frame01)
        frame21 = to_variable(frame21)
        tensorDot11 = sepconv.sepconv_func.apply(self.modulePad(frame01), Vertical1, Horizontal1)
        tensorDot21 = sepconv.sepconv_func.apply(self.modulePad(frame21), Vertical2, Horizontal2)

        # 核归一化
        frame1 = (tensorDot1 + tensorDot2) / (tensorDot11 + tensorDot21)
        # 反归一化
        # frame1 = frame1*torch.std(mixframe)+torch.mean(mixframe)

        # 恢复到填充之前的大小
        if h_padded:
            frame1 = frame1[:, :, 0:h0, :]
        if w_padded:
            frame1 = frame1[:, :, :, 0:w0]

        return frame1

    def train_model(self, frame0, frame2, frame1):
        self.optimizer.zero_grad()
        # 对输入两帧图像做联合归一化
        # mixframe = torch.cat([frame0, frame2], dim=2)
        # frame0 = (frame0-torch.mean(mixframe))/torch.std(mixframe)
        # frame2 = (frame2 - torch.mean(mixframe)) / (torch.std(mixframe)+0.0000001)
        output = self.forward(frame0, frame2)
        # 反归一化
        # frame1 = frame1*torch.std(mixframe)+torch.mean(mixframe)

        # loss = 0.3*self.criterion(modelvgg19(output), modelvgg19(frame1)) + 0.7*self.criterion(output, frame1)
        # loss = self.criterion(output, frame1) + (-pytorch_ssim.SSIM(frame1, output))
        loss = self.criterion(output, frame1)
        loss.backward()
        self.optimizer.step()
        return loss

    def increase_epoch(self):
        self.epoch += 1
        # self.scheduler.step()
