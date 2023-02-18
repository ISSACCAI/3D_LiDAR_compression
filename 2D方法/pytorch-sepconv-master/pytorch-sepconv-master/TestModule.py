from PIL import Image
import torch
from torchvision import transforms
from math import log10
from torchvision.utils import save_image as imwrite
from torch.autograd import Variable
import os
import pytorch_ssim

my_result_txt= "my_result.txt"


val_list = []
for i in range(1344):  # 1340
    val_list.append(str(i+1500))


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class Middlebury_eval:
    def __init__(self, input_dir):
        self.im_list = ['Army', 'Backyard', 'Basketball', 'Dumptruck', 'Evergreen', 'Grove', 'Mequon', 'Schefflera',
                        'Teddy', 'Urban', 'Wooden', 'Yosemite']


class Middlebury_other:
    def __init__(self, input_dir, gt_dir):
        # self.im_list = ['2011_09_26_drive_0002_sync', '2011_09_26_drive_0005_sync', '2011_09_26_drive_0017_sync', '2011_09_26_drive_0018_sync']
        self.im_list = val_list
        # self.im_list = ['00', '1', '100', '200', '300', '400', '500', '600', '700', '800', '900']
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        for item in self.im_list:
            self.input0_list.append(
                to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame10.png')).unsqueeze(0)))
            self.input1_list.append(
                to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame11.png')).unsqueeze(0)))
            # 以下因为要计算程序运行时间所以注释
            # self.gt_list.append(
            #     to_variable(self.transform(Image.open(gt_dir + '/' + item + '/frame10i11.png')).unsqueeze(0)))

    def Test(self, model, output_dir, logfile=None, output_name='output.png'):
        # 以下因为要计算程序运行时间所以注释
        # av_psnr = 0
        # av_ssim = 0
        # if logfile is not None:
        #     logfile.write('{:<7s}{:<3d}'.format('Epoch: ', model.epoch.item()) + '\n')
        for idx in range(len(self.im_list)):
            if not os.path.exists(output_dir + '/' + self.im_list[idx]):
                os.makedirs(output_dir + '/' + self.im_list[idx])
            self.input0_list[idx] = self.input0_list[idx].type(torch.cuda.FloatTensor)
            self.input1_list[idx] = self.input1_list[idx].type(torch.cuda.FloatTensor)
            frame_out = model(self.input0_list[idx], self.input1_list[idx])
            # 以下因为要计算程序运行时间所以注释
        #     gt = self.gt_list[idx]
        #
        #     ssim_value = pytorch_ssim.ssim(gt, frame_out).item()
        #     av_ssim += ssim_value
        #
        #     psnr = -10 * log10(torch.mean((gt - frame_out) * (gt - frame_out)).item())
        #     av_psnr += psnr
        #     imwrite(frame_out, output_dir + '/' + self.im_list[idx] + '/' + output_name, range=(0, 1))
        #     msg = '{:<15s}{:<20.16f}'.format(self.im_list[idx] + ': ', psnr) + '\n'
        #     print(msg, end='')
        #     if logfile is not None:
        #         logfile.write(msg)
        #     print(self.im_list[idx] + "psnr的结果为：" + str(psnr))
        #     print(self.im_list[idx] + "ssim的结果为：" + str(ssim_value))
        #     with open(my_result_txt, "a+") as f:
        #         f.write(self.im_list[idx] + "psnr的结果为：" + str(psnr) + "\n")
        #         f.write(self.im_list[idx] + "ssim的结果为：" + str(ssim_value) + "\n")
        # av_psnr /= len(self.im_list)
        # msg = '{:<15s}{:<20.16f}'.format('Average: ', av_psnr) + '\n'
        # print(msg, end='')
        # if logfile is not None:
        #     logfile.write(msg)
        # print("total_psnr__average:" + str(av_psnr))
        # av_ssim /= len(self.im_list)
        # print("total_ssim_average:" + str(av_ssim))
        # with open(my_result_txt, "a+") as f:
        #     f.write("total_PSNR_average:" + str(av_psnr))
        #     f.write("total_SSIM_average:" + str(av_ssim))


#只用了一半测试集，要修改。