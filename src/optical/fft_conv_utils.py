from typing import Union
from functools import partial
import torch
import torch.fft
import torch.nn.functional as F
import numpy as np
import math
import cv2
from numpy import fft


def prepare_kernel(image_size):
    MAX_GRAY_VALUE = 255.    # 定义最大灰度值常量，用于归一化

    kernel = cv2.imread('conv_kernel_960_640.jpg')      # 'conv_kernel_960_640.jpg' 是预先设计或估计的模糊核可视化图像   后面可变
    kernel = cv2.resize(kernel, image_size)   # 将核图像调整到指定尺寸，使其与目标图像尺寸匹配
    kernel = np.flip(np.flip(kernel, axis=0), axis=1)
    # 在垂直和水平方向翻转核图像
    # 这是因为卷积操作需要翻转核，而OpenCV的滤波函数会自动执行这个操作
    # 但在这里我们手动进行翻转，是为了与特定的卷积实现方式配合
    kernel = cv2.cvtColor(kernel, cv2.COLOR_BGR2RGB) / MAX_GRAY_VALUE
    # 将图像从BGR格式转换为RGB格式，并归一化到[0,1]范围
    kernel = torch.from_numpy(kernel[np.newaxis].transpose(0, 3, 1, 2)).float()  # 将numpy数组转换为PyTorch张量，并调整维度顺序
    k_row_ori, k_col_ori = kernel.shape[2:4]       # 获取核的原始高度和宽度
    if k_row_ori % 2 == 0 and k_col_ori % 2 == 0:       # 确保核的尺寸为奇数，这对于某些卷积操作是必要的
        kernel = F.pad(kernel, (0, 1, 0, 1))       # 如果高度和宽度都是偶数，则在右下各填充1个像素
    elif k_row_ori % 2 == 0:
        kernel = F.pad(kernel, [0, 0, 0, 1])
    elif k_col_ori % 2 == 0:
        kernel = F.pad(kernel, [0, 1, 0, 0])
    k_row, k_col = kernel.shape[2:4]
    kernel_size_half = (math.floor(k_row / 2), math.floor(k_col / 2))

    return kernel, kernel_size_half

def pad_kernel(kernel):     #填充
    k_row_ori, k_col_ori = kernel.shape[2:4]
    if k_row_ori % 2 == 0 and k_col_ori % 2 == 0:
        kernel = F.pad(kernel, (0, 1, 0, 1))
    elif k_row_ori % 2 == 0:
        kernel = F.pad(kernel, [0, 0, 0, 1])
    elif k_col_ori % 2 == 0:
        kernel = F.pad(kernel, [0, 1, 0, 0])
    k_row, k_col = kernel.shape[2:4]
    kernel_size_half = (math.floor(k_row / 2), math.floor(k_col / 2))

    return kernel, kernel_size_half

# 3. Multiply the transformed matrices
def complex_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiplies two complex-valued tensors."""
    # Scalar matrix multiplication of two tensors, over only the first two dimensions.
    # Dimensions 3 and higher will have the same shape after multiplication.
    scalar_matmul = partial(torch.einsum, "ab..., cb... -> ac...")

    # Compute the real and imaginary parts independently, then manually insert them
    # into the output Tensor.  This is fairly hacky but necessary for PyTorch 1.7.0,
    # because Autograd is not enabled for complex matrix operations yet.  Not exactly
    # idiomatic PyTorch code, but it should work for all future versions (>= 1.7.0).
    real = scalar_matmul(a.real, b.real) - scalar_matmul(a.imag, b.imag)
    imag = scalar_matmul(a.imag, b.real) + scalar_matmul(a.real, b.imag)
    c = torch.complex(real, imag)
    # c = torch.zeros(real.shape, dtype=torch.complex64, device=a.device)
    # c.real, c.imag = real, imag

    return c


def fft_conv_1d(
        signal: torch.Tensor, kernel: torch.Tensor, bias: torch.Tensor = None,
        padding: int = 0) -> torch.Tensor:
    """
    Args:
        signal: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Optional, Tensor) Bias tensor to add to the output.
        padding: (int) Number of zero samples to pad the input on the last dimension.
    Returns:
        (Tensor) Convolved tensor
    """
    # 1. Pad the input signal & kernel tensors
    signal = F.pad(signal, [padding, padding])
    kernel_padding = [0, signal.size(-1) - kernel.size(-1)]
    padded_kernel = F.pad(kernel, kernel_padding)

    # 2. Perform fourier convolution
    signal_fr = torch.fft.rfftn(signal, dim=-1)
    kernel_fr = torch.fft.rfftn(padded_kernel, dim=-1)

    # 3. Multiply the transformed matrices
    kernel_fr.imag *= -1
    output_fr = complex_matmul(signal_fr, kernel_fr)

    # 4. Compute inverse FFT, and remove extra padded values
    output = torch.fft.irfftn(output_fr, dim=-1)
    output = output[:, :, :signal.size(-1) - kernel.size(-1) + 1]

    # 5. Optionally, add a bias term before returning.
    if bias is not None:
        output += bias.view(1, -1, 1)

    return output


def fft_conv_2d(
        signal: torch.Tensor, kernel: torch.Tensor, bias: torch.Tensor = None,
        padding: Union[int, int] = (0, 0), image_shift=None) -> torch.Tensor:
    """
    Args:
        signal: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Optional, Tensor) Bias tensor to add to the output.
        padding: (int) Number of zero samples to pad the input on the last dimension.
    Returns:
        (Tensor) Convolved tensor
    """
    signal_row_ori, signal_col_ori = signal.shape[2:4]
    if image_shift is not None:
        if image_shift[0] >= 0:
            signal = F.pad(signal, [0, 0, image_shift[0], 0])
        elif image_shift[0] < 0:
            kernel = F.pad(kernel, [0, 0, -image_shift[0], 0])

        if image_shift[1] >= 0:
            signal = F.pad(signal, [image_shift[1], 0, 0, 0])
        elif image_shift[1] < 0:
            kernel = F.pad(kernel, [-image_shift[1], 0, 0, 0])

    # 1. Pad the input signal & kernel tensors
    signal = F.pad(signal, [padding[1], padding[1], padding[0], padding[0]])
    kernel_padding = [0, signal.size(-1) - kernel.size(-1),
                      0, signal.size(-2) - kernel.size(-2)]
    padded_kernel = F.pad(kernel, kernel_padding)

    # 2. Perform fourier convolution
    signal_fr = torch.fft.rfftn(signal, dim=[-2, -1])
    kernel_fr = torch.fft.rfftn(padded_kernel, dim=[-2, -1])

    # 3. Multiply the transformed matrices
    kernel_fr.imag *= -1
    output_fr = complex_matmul(signal_fr, kernel_fr)

    # 4. Compute inverse FFT, and remove extra padded values
    output = torch.fft.irfftn(output_fr, dim=[-2, -1])


    if image_shift is not None:
        if image_shift[0] >= 0:
            output = output[:, :, :signal_row_ori, :]
        else:
            output = output[:, :, -image_shift[0]:-image_shift[0] + signal_row_ori, :]
        if image_shift[1] >= 0:
            output = output[:, :, :, :signal_col_ori]
        else:
            output = output[:, :, :, -image_shift[1]:-image_shift[1] + signal_col_ori]
    else:
        output = output[:, :, :signal_row_ori, :signal_col_ori]

    # 5. Optionally, add a bias term before returning.
    if bias is not None:
        output += bias.view(1, -1, 1, 1)

    return output


def shift_kernel(kernel, image_shift):
    # 注意这里的image_shift是指的被卷积图像应当进行的偏移,而不是kernel应当进行的偏移
    row_ori, col_ori = kernel.shape[2:4]
    if image_shift[0] >= 0:
        kernel = kernel[:, :, image_shift[0]:]
        kernel = F.pad(kernel, [0, 0, 0, image_shift[0]])
    elif image_shift[0] < 0:
        kernel = F.pad(kernel, [0, 0, -image_shift[0], 0])
        kernel = kernel[:, :, :row_ori]
    if image_shift[1] >= 0:
        kernel = kernel[:, :, :, image_shift[1]:]
        kernel = F.pad(kernel, [0, image_shift[1], 0, 0])
    elif image_shift[1] < 0:
        kernel = F.pad(kernel, [-image_shift[1], 0, 0, 0])
        kernel = kernel[:, :, :, :col_ori]

    return kernel


# 对图片进行模糊
def make_blurred(input, PSF, eps):
    input_fft = fft.fft2(input)  # 进行二维数组的傅里叶变换
    PSF_fft = fft.fft2(PSF) + eps
    blurred = fft.ifft2(input_fft * PSF_fft)
    blurred = np.abs(fft.fftshift(blurred))
    return blurred


def wiener(input, PSF, eps, K=0.01):  # 维纳滤波,K=0.01
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) + eps
    PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)
    result = fft.ifft2(input_fft * PSF_fft_1)
    result = np.abs(fft.fftshift(result))
    return result


def get_wn_fun(image, kernel, K_value=None):
    if K_value is None:
        K_value = 0.005
    # 对暗图像执行维纳滤波
    wn_list = []
    for channel_id in range(kernel.shape[1]):
        blurred_now = image[:, :, channel_id]
        PSF = kernel[0, channel_id].numpy()
        result_wn_now = wiener(blurred_now, PSF, 1e-3, K=K_value)  # 维纳滤波
        result_wn_now = np.maximum(0, np.minimum(255, result_wn_now))
        wn_list.append(result_wn_now[:, :, np.newaxis])

    wn_result = np.concatenate(wn_list, axis=2)
    wn_result = np.rint(wn_result).astype('uint8')
    return wn_result
