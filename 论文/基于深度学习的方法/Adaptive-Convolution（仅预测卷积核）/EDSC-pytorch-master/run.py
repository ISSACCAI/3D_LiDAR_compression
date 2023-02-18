import cv2
import torch
import numpy as np
import math
from networks import EDSC
import getopt
import sys
import os
from torchvision import transforms
from torch.autograd import Variable
from math import log10
import pytorch_ssim
import PIL

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

assert (int(str('').join(torch.__version__.split('.')[0:3])) >= 100)  # requires at least pytorch version 1.0.0

torch.set_grad_enabled(False)  # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

arguments_strModel = "EDSC_s"
arguments_strModelStateDict = './EDSC_s_l1.ckpt'

arguments_strFirst = './frame10.png'
arguments_strSecond = './frame11.png'
arguments_strOut = './out.png'
arguments_intDevice = 0
arguments_floatTime = 0.1

for strOption, strArgument in \
getopt.getopt(sys.argv[1:], '', [strParameter[2:] + '=' for strParameter in sys.argv[1::2]])[0]:
    if strOption == '--device' and strArgument != '': arguments_intDevice = int(strArgument)  # device number
    if strOption == '--model' and strArgument != '': arguments_strModel = strArgument  # model type
    if strOption == '--model_state' and strArgument != '': arguments_strModelStateDict = strArgument  # path to the model state
    if strOption == '--first' and strArgument != '': arguments_strFirst = strArgument  # path to the first frame
    if strOption == '--second' and strArgument != '': arguments_strSecond = strArgument  # path to the second frame
    if strOption == '--out' and strArgument != '': arguments_strOut = strArgument  # path to where the output should be stored
    if strOption == '--time' and strArgument != '': arguments_floatTime = float(strArgument)  # the intermediate time of the synthesized frame

torch.cuda.set_device(arguments_intDevice)


def evaluate(im1_path, im2_path, save_path):
    if arguments_strModel == "EDSC_s":
        GenerateModule = EDSC.Network(isMultiple=False).cuda()
        GenerateModule.load_state_dict(
            torch.load(arguments_strModelStateDict, map_location=lambda storage, loc: storage)['model_state'])
        GenerateModule.eval()

    elif arguments_strModel == "EDSC_m":
        GenerateModule = EDSC.Network(isMultiple=True).cuda()
        GenerateModule.load_state_dict(
            torch.load(arguments_strModelStateDict, map_location=lambda storage, loc: storage)['model_state'])
        GenerateModule.eval()

    with torch.no_grad():
        path1 = im1_path
        path2 = im2_path

        write_path = save_path
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)

        assert img1.shape == img2.shape

        temp_input_images1 = np.zeros((1, img1.shape[0], img1.shape[1], img1.shape[2]), np.float32)
        temp_input_images2 = np.zeros((1, img1.shape[0], img1.shape[1], img1.shape[2]), np.float32)

        temp_input_images1[0, :, :, :] = img1[:, :, :].astype(np.float32) / 255.0
        temp_input_images2[0, :, :, :] = img2[:, :, :].astype(np.float32) / 255.0

        temp_input_images1 = np.rollaxis(temp_input_images1, 3, 1)
        temp_input_images2 = np.rollaxis(temp_input_images2, 3, 1)

        img1_V = torch.from_numpy(temp_input_images1).cuda()
        img2_V = torch.from_numpy(temp_input_images2).cuda()

        modulePaddingInput = torch.nn.ReplicationPad2d(
            [0, int((math.ceil(img1_V.size(3) / 32.0) * 32 - img1_V.size(3))), 0,
             int((math.ceil(img1_V.size(2) / 32.0) * 32 - img1_V.size(2)))])
        modulePaddingOutput = torch.nn.ReplicationPad2d(
            [0, 0 - int((math.ceil(img1_V.size(3) / 32.0) * 32 - img1_V.size(3))), 0,
             0 - int((math.ceil(img1_V.size(2) / 32.0) * 32 - img1_V.size(2)))])

        img1_V_padded = modulePaddingInput(img1_V)
        img2_V_padded = modulePaddingInput(img2_V)

        if arguments_strModel == 'EDSC_s':
            variableOutput = GenerateModule([img1_V_padded, img2_V_padded])
            variableOutput = modulePaddingOutput(variableOutput)
        elif arguments_strModel == 'EDSC_m':
            time_torch = torch.ones((1, 1, int(img1_V_padded.shape[2] / 2), int(img1_V_padded.shape[3] / 2))) * arguments_floatTime
            variableOutput = GenerateModule([img1_V_padded, img2_V_padded, time_torch.cuda()])
            variableOutput = modulePaddingOutput(variableOutput)

        output = variableOutput.data.permute(0, 2, 3, 1)
        out = output.cpu().clamp(0.0, 1.0).numpy() * 255.0
        result = out.squeeze().astype(np.uint8)
        cv2.imwrite(write_path, result)

    return result

def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

if __name__ == '__main__':
    revisting_result_txt = "EDEC_result.txt"
    av_psnr = 0
    av_ssim = 0
    input_dir = "C:/Users/Cauli/Desktop/db_test/input/"
    gt_dir = "C:/Users/Cauli/Desktop/db_test/gt/"
    output_dir = "C:/Users/Cauli/Desktop/result/muli_enhanced_sepconvlution/"
    path_list = os.listdir(input_dir)

    path_add_output = []

    filename_list = []
    path_add_gt_ = []
    path_add_input1 = []
    path_add_input2 = []
    for filename in path_list:
        path_add_input = input_dir + filename + '/'

        input_list = os.listdir(path_add_input)
        path_add_input1.append(path_add_input + input_list[0])
        path_add_input2.append(path_add_input + input_list[1])
        path_add_gt = gt_dir + '/' + filename + '/'
        gt_list = os.listdir(path_add_gt)
        path_add_gt_.append(path_add_gt + gt_list[0] + '/')
        path_add_list = output_dir + filename + '/'
        path_add_output.append(path_add_list + "out.png")

        filename_list.append(filename)

    for i in range(len(path_list)):
        evaluate(path_add_input1[i], path_add_input2[i], path_add_output[i])

        transform = transforms.Compose([transforms.ToTensor()])

        frame_out = to_variable(transform(PIL.Image.open(path_add_output[i]))).unsqueeze(0)

        gt = to_variable(transform(PIL.Image.open(path_add_input2[i]))).unsqueeze(0)

        ssim_value = pytorch_ssim.ssim(gt, frame_out).item()
        av_ssim += ssim_value
        psnr = -10 * log10(torch.mean((gt - frame_out) * (gt - frame_out)).item())
        av_psnr += psnr
        print(filename_list[i] + "psnr的结果为：" + str(psnr))
        print(filename_list[i] + "ssim的结果为：" + str(ssim_value))
        with open(revisting_result_txt, "a+") as f:
            f.write(filename_list[i] + "psnr的结果为：" + str(psnr) + "\n")
            f.write(filename_list[i] + "ssim的结果为：" + str(ssim_value) + "\n")

    av_psnr /= len(path_add_input1)
    print("total_psnr__average:" + str(av_psnr))
    av_ssim /= len(path_add_input1)
    print("total_ssim_average:" + str(av_ssim))
    with open(revisting_result_txt, "a+") as f:
        f.write("total_PSNR_average:" + str(av_psnr))
        f.write("total_SSIM_average:" + str(av_ssim))
