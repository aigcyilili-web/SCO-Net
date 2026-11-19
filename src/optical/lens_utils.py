import numpy as np
import math
import torch
import torch.fft as fft
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os.path as osp

def get_init_height(wave_resolution):

    # 待确认，这里需要确认x对应wave_resolution[0]还是wave_resolution[1]
    x0=np.linspace(-wave_resolution[0] * 3 / 2 * 1e-6, 
                   wave_resolution[0] * 3 / 2 * 1e-6, wave_resolution[0])
    y0=np.linspace(-wave_resolution[1] * 3 / 2 * 1e-6, 
                   wave_resolution[1] * 3 / 2 * 1e-6, wave_resolution[1])
    [xx,yy]=np.meshgrid(x0,y0)

    t_460 = -213423913 * (xx*xx+yy*yy)#菲涅尔透镜(460-341478261)(550-285600000)(640-245436924)
    t_550 = -153300000 * (xx*xx+yy*yy)
    t_640 = -153398438 * (xx*xx+yy*yy)

    t_460 = torch.tensor(t_460)

    #以菲涅尔透镜的高度分布作为后续训练的初始高度
    height_460 = torch.angle(torch.exp(1j * t_460)) / ((2. * np.pi / (550 * 1e-9))*(0.7091))

    # phase_height = np.zeros([1,wave_resolution[0],wave_resolution[1],3])
    # phase_height[0,:,:,0] = height_460 
    # phase_height[0,:,:,1] = height_460
    # phase_height[0,:,:,2] = height_460
    # init_height = torch.tensor(phase_height, dtype=torch.float32)
    
    init_height = height_460.float()

    init_height = init_height - torch.min(init_height)
    
    # max_init_height=1e-10
    # max_height_now=torch.max(init_height)
    # if max_height_now>max_init_height:
    #     init_height=init_height*(max_init_height/max_height_now)
    return init_height

def get_init_cos_height(wave_resolution):
    # 待确认，这里需要确认x对应wave_resolution[0]还是wave_resolution[1]
    x0 = np.linspace(-wave_resolution[0] * 1 / 2,
                     wave_resolution[0] * 1 / 2 , wave_resolution[0])
    y0 = np.linspace(-wave_resolution[1] * 1 / 2 ,
                     wave_resolution[1] * 1 / 2 , wave_resolution[1])
    [xx, yy] = np.meshgrid(x0, y0)

    resolution = 960
    freq = 350  # 选择的空间频率
    freq1 = 400
    freq2 = 450

    phase_mask = np.cos(2 * np.pi * freq / resolution * xx) + np.cos(2 * np.pi * freq1 / resolution * yy) + \
                 np.cos(2 * np.pi * freq2 / resolution * (xx + yy))

    phase_mask = np.mod(phase_mask, 2 * np.pi)
    phase_mask = torch.tensor(phase_mask)

    # 以菲涅尔透镜的高度分布作为后续训练的初始高度
    phase_mask = torch.angle(torch.exp(1j * phase_mask)) / ((2. * np.pi / (550 * 1e-9)) * (0.7091))



    init_height = phase_mask.float()

    init_height = init_height - torch.min(init_height)
    return init_height


def compl_exp_torch(phase, dtype=torch.complex64):
    """Complex exponent via euler's formula, since CUDA doesn't have a GPU kernel for that.
    Casts to *dtype*.
    """
    assert phase.dtype==torch.float32
    real = torch.cos(phase).to(dtype)
    imag = torch.sin(phase).to(dtype)
    return torch.add(real, 1j * imag)

def get_spherical_wavefront_phase(resolution, physical_size, wave_lengths, source_distance):
    source_distance = source_distance.to(torch.float64)
    physical_size = physical_size.to(torch.float64)
    wave_lengths = wave_lengths.to(torch.float64)

    N, M = resolution
    x_ori, y_ori = torch.meshgrid(torch.arange(-N // 2, N // 2),
                          torch.arange(-M // 2, M // 2))
    x, y = torch.meshgrid(torch.arange(-N // 2, N // 2),
                          torch.arange(-M // 2, M // 2),indexing='ij')
    
    x = x.to(torch.float64) / N * physical_size
    y = y.to(torch.float64) / M * physical_size

    curvature = torch.sqrt(x ** 2 + y ** 2 + source_distance ** 2)
    wave_nos = 2. * np.pi / wave_lengths

    phase_shifts = compl_exp_torch(wave_nos * curvature)

    phase_shifts = phase_shifts.unsqueeze(0).unsqueeze(-1)
    return phase_shifts

def circular_aperture(input_field):
    input_shape = input_field.shape
    x, y = torch.meshgrid(torch.arange(-input_shape[1] // 2, input_shape[1] // 2),
                          torch.arange(-input_shape[2] // 2, input_shape[2] // 2),
                          indexing='ij')
    max_val = torch.amax(x)

    r = torch.sqrt(x ** 2 + y ** 2).unsqueeze(0).unsqueeze(-1)
    aperture = (r < max_val).to(torch.float32).to(input_field.device)
    return aperture * input_field

def PhasePlate(input_field, phase_shifts):
    assert input_field.dtype==torch.complex64
    assert phase_shifts.dtype==torch.complex64
    
    return input_field * phase_shifts

def ifftshift(x, axes=None):
    x = torch.as_tensor(x)
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(axes, int):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[ax] // 2) for ax in axes]

    return torch.roll(x, shift, dims=axes)

def transp_fft2d(a_tensor):
    assert a_tensor.dtype==torch.complex64
    a_tensor_transp = a_tensor.permute(0, 3, 1, 2)
    a_fft2d = fft.fft2(a_tensor_transp)
    a_fft2d = a_fft2d.permute(0, 2, 3, 1)
    return a_fft2d

def transp_ifft2d(a_tensor):
    assert a_tensor.dtype==torch.complex64
    a_tensor = a_tensor.permute(0, 3, 1, 2)
    a_ifft2d_transp = fft.ifft2(a_tensor)
    a_ifft2d = a_ifft2d_transp.permute(0, 2, 3, 1)
    return a_ifft2d

def get_intensities(input_field):
    return torch.square(torch.abs(input_field))

def ifftshift2d_torch(a_tensor):
    input_shape = a_tensor.shape

    new_tensor = a_tensor
    for axis in range(1, 3):
        n = input_shape[axis]
        split = n - (n + 1) // 2
        mylist = np.concatenate((np.arange(split, n), np.arange(split)))
        new_tensor = new_tensor.index_select(axis, torch.tensor(mylist))
    return new_tensor

def psf2otf(input_filter, output_size):
    fh, fw, _, _ = input_filter.shape

    if output_size[0] != fh:
        pad = (output_size[0] - fh) / 2

        if (output_size[0] - fh) % 2 != 0:
            pad_top = pad_left = math.ceil(pad)
            pad_bottom = pad_right = math.floor(pad)
        else:
            pad_top = pad_left = int(pad) + 1
            pad_bottom = pad_right = int(pad) - 1
        padded = F.pad(input_filter, (0, 0, 0, 0, pad_left, pad_right, pad_top, pad_bottom))
    else:
        padded = input_filter

    padded = padded.permute(2, 0, 1, 3)
    padded = ifftshift2d_torch(padded)
    padded = padded.permute(1, 2, 0, 3)

    tmp = padded.permute(2, 3, 0, 1)
    tmp = fft.fft2(torch.complex(tmp, torch.tensor(0.)))

    return tmp.permute(2, 3, 0, 1)

def img_psf_conv(img, psf, show_figure_mark=False):
    assert img.dtype==torch.float32
    assert psf.dtype==torch.float32

    img_shape = img.shape
    
    target_side_length = 2 * img_shape[1]
    height_pad = (target_side_length - img_shape[1]) / 2
    width_pad = (target_side_length - img_shape[1]) / 2

    pad_top, pad_bottom = int(math.ceil(height_pad)), int(math.floor(height_pad))
    pad_left, pad_right = int(math.ceil(width_pad)), int(math.floor(width_pad))
    img = F.pad(img, (0, 0, pad_left, pad_right, pad_top, pad_bottom))
    img_shape = img.shape

#     print("img.size:",img.size())#([1, 3744, 3744, 3])
    
    img_fft = transp_fft2d(img.to(torch.complex64))
#     print("img_fft.size:",img_fft.size())#([1, 3744, 3744, 3])

    otf = psf2otf(psf, output_size=img_shape[1:3])#img_shape[2:4]
#     print("otf.size:",otf.size())#([3744, 3744, 1, 3])
    
    otf = otf.permute(2, 0, 1, 3)
#     print("otf.size2:",otf.size())#([1, 3744, 3744, 3])
    
    assert otf.dtype==torch.complex64
    assert img_fft.dtype==torch.complex64
    
    result = transp_ifft2d(img_fft * otf)
#     print("result.size:",result.size())#([1, 3744, 3744, 3])
    
    if show_figure_mark:
        plt.subplot(1,3,1)
        plt.imshow(torch.abs(result[0,:,:,0]),vmax=0.005)
        plt.colorbar()
        plt.subplot(1,3,2)
        plt.imshow(torch.abs(result[0,:,:,1]),vmax=0.005)
        plt.colorbar()
        plt.subplot(1,3,3)
        plt.imshow(torch.abs(result[0,:,:,2]),vmax=0.005)
        plt.colorbar()
        plt.show()
    
    result = result.to(torch.float32)
    result = result[:, pad_top:-pad_bottom, pad_left:-pad_right, :]

    return result

def gaussian_noise(image, stddev=0.001):
    dtype = image.dtype
    return image + torch.randn(image.shape, dtype=dtype) * stddev

def least_common_multiple(a, b):
    return abs(a * b) // math.gcd(a, b) if a and b else 0

def area_downsampling_torch(input_image, target_side_length):
    input_shape = input_image.shape
    input_image = input_image.to(torch.float32)

    if input_shape[2] % target_side_length == 0:
        factor = int(input_shape[2] / target_side_length)
        output_img = F.avg_pool2d(input_image, factor)
    else:
        lcm_factor = least_common_multiple(target_side_length, input_shape[2]) / target_side_length

        if lcm_factor > 10:
            print("Warning: area downsampling is very expensive and not precise if source and target wave length have a large least common multiple")
            upsample_factor = 10
        else:
            upsample_factor = int(lcm_factor)

        img_upsampled = F.interpolate(input_image, scale_factor=upsample_factor, mode='nearest')
        output_img = F.avg_pool2d(img_upsampled, upsample_factor)

    return output_img

def inverse_filter(blurred, estimate, psf, gamma=None, init_gamma=2.):
    img_shape = list(blurred.size())

    if gamma is None:
        # Gamma (the regularization parameter) is also a trainable parameter.
        gamma = torch.tensor(init_gamma, dtype=torch.float32)
        gamma = torch.square(gamma)  # Enforces positivity of gamma.

    a_tensor_transp = blurred.permute(0, 3, 1, 2)
    estimate_transp = estimate.permute(0, 3, 1, 2)

    # Everything has shape (batch_size, num_channels, height, width)
    img_fft = fft.fftn(torch.complex(a_tensor_transp, torch.zeros_like(a_tensor_transp)))
    otf = psf2otf(psf, output_size=img_shape[2:4])
    otf = otf.permute(2, 3, 0, 1)

    adj_conv = img_fft * torch.conj(otf)

    # This is a slight modification to standard inverse filtering - gamma not only regularizes the inverse filtering,
    # but also trades off between the regularized inverse filter and the unfiltered estimate_transp.
    numerator = adj_conv + fft.fftn(torch.complex(gamma * estimate_transp, torch.zeros_like(estimate_transp)))

    kernel_mags = torch.square(torch.abs(otf))  # Magnitudes of the blur kernel.

    denominator = torch.complex(kernel_mags + gamma, torch.zeros_like(kernel_mags))
    filtered = numerator / denominator
    cplx_result = fft.ifftn(filtered)
    real_result = torch.real(cplx_result)  # Discard complex parts.

    # Get back to (batch_size, num_channels, height, width)
    result = real_result.permute(0, 2, 3, 1)
    result = 1. / torch.max(result) * result

    return result

def phaseshifts_from_height_map(height_map, wave_lengths, refractive_idcs):
    '''Calculates the phase shifts created by a height map with certain
    refractive index for light with specific wave length.
    '''
    # refractive index difference
    delta_N = refractive_idcs.reshape((1, 1, 1, -1)) - 1.
    # wave number
    wave_nos = 2. * np.pi / wave_lengths
    wave_nos = wave_nos.reshape((1, 1, 1, -1))
    # phase delay indiced by height field
    phi = wave_nos * delta_N * height_map
    phase_shifts = compl_exp_torch(phi)
    return phase_shifts

def Fresnelpropagation(input_field, input_sample_interval, distance, wave_lengths):
    input_shape = input_field.shape
    _, _, N_orig, _ = input_shape

    Npad = N_orig // 6
    
    N = N_orig + 2 * Npad
#     print("input_field.size:", input_field.size())
    padded_input_field = F.pad(input_field, (0, 0, Npad, Npad, Npad, Npad))
#   
    [x, y] = np.mgrid[-N // 2:N // 2, -N // 2:N // 2]

    fx = x / (input_sample_interval * N)
    fy = y / (input_sample_interval * N)

    fx = ifftshift(fx.astype(np.float32))
    fy = ifftshift(fy.astype(np.float32))
    fx = fx.unsqueeze(-1).unsqueeze(0)
    fy = fy.unsqueeze(-1).unsqueeze(0)

    squared_sum = torch.square(fx) + torch.square(fy)
    tmp = wave_lengths * torch.tensor(np.pi * -1.,dtype=torch.float32)* squared_sum
    assert tmp.dtype==torch.float32
    H = compl_exp_torch(distance * tmp, dtype=torch.complex64)
    H = H.to(input_field.device)
#     print("padded_input_field.size:", padded_input_field.size())
    objFT = transp_fft2d(padded_input_field)
#     print("objFT size:",objFT.size())
#     print("H:",H.size())
    out_field = transp_ifft2d(objFT * H)

    return out_field[:, Npad:-Npad, Npad:-Npad, :]

def get_psfs(distance, sensor_distance, phaplate_distance, wave_resolution, physical_size, 
             wave_lengths, phase1, phase2, input_sample_interval, kernel_size, point_light_shift_xy):
    device=phase1.device
    x0,y0=point_light_shift_xy
    
    N, M = wave_resolution
    [y, x] = np.mgrid[-N//2:N//2, -M//2:M//2].astype(np.float32)
    x=x+x0
    y=y+y0
    x = x / N * physical_size
    y = y / N * physical_size

    squared_sum = x ** 2 + y ** 2
    wave_num = 2. * np.pi / wave_lengths
    wave_num = wave_num.reshape([1, 1, 1, -1])
    curvature = torch.sqrt(torch.tensor(squared_sum, dtype=torch.float32)\
                + torch.tensor(distance, dtype=torch.float32) ** 2) \
                -torch.tensor(distance, dtype=torch.float32)
    curvature = curvature.unsqueeze(0).unsqueeze(-1)
    spherical_wavefront = compl_exp_torch(wave_num * curvature)

    # # 调试使用,检查点光源球面波的相位图
    # angle_1 = torch.angle(spherical_wavefront)
    # angle_1=angle_1.cpu().numpy()
    # for channel_id in range(3):
    #     plt.imshow(angle_1[0,:,:,channel_id])
    #     plt.colorbar()
    #     plt.savefig(osp.join('debug_image', 'angle1_%d.png'%channel_id))
    #     plt.clf()
    #     plt.close()

    input_field = spherical_wavefront
    
    field1 = PhasePlate(input_field.to(device), phase1)
    field1 = circular_aperture(field1)
    field1 = Fresnelpropagation(field1, input_sample_interval, phaplate_distance, wave_lengths)
    field2 = PhasePlate(field1, phase2)
    field2 = circular_aperture(field2)
    sensor_incident_field = Fresnelpropagation(field2, input_sample_interval, sensor_distance, wave_lengths)
    psf = get_intensities(sensor_incident_field)
    
    # This permuting operation finish the processes of transposing and switching channels
    # psf = psf.transpose(psf, 1, 2)
    psf = psf.permute((0,3,2,1))
    
    # import cv2
    # for sele_pos in range(3):
    #     kernel_np=psf[0,sele_pos,:,:].detach().cpu().numpy()
    #     kernel_np=np.rint((kernel_np-np.min(kernel_np))/(np.max(kernel_np)-np.min(kernel_np))*255).astype('uint8')
    #     cv2.imwrite('debug_image/kernel_%d.bmp'%sele_pos, kernel_np)

    # resize to the desired image shape
    psf = F.interpolate(psf, size=kernel_size, mode='bilinear', align_corners=False)
    
    # 中文：待确认，是否需要除以torch.sum(psf, dim=[1, 2])
    psf = psf/torch.sum(psf, dim=[2,3], keepdim=True)
    return psf

def get_sensor_img(input_img, wave_resolution, psfs, patch_size, 
                   noise_sigma = 0.001, show_figure_mark=False):
    
    input_img = torch.permute(input_img, [0,3,1,2])
    input_img = F.interpolate(input_img, size=wave_resolution, mode='nearest')
    input_img = torch.permute(input_img, [0,2,3,1])
    sensor_img = img_psf_conv(input_img, psfs, show_figure_mark)
    sensor_img = gaussian_noise(sensor_img, noise_sigma)
    sensor_img = area_downsampling_torch(sensor_img, patch_size)

    return sensor_img

def shift_tensor(kernel, shift_rc):
    if shift_rc[0]==0 and shift_rc[1]==0:
        return kernel
    
    row_ori, col_ori = kernel.shape[2:4]
    if shift_rc[0] >= 0:
        kernel = F.pad(kernel, [0, 0, shift_rc[0], 0])
        kernel = kernel[:, :, :row_ori]
    elif shift_rc[0] < 0:
        kernel = kernel[:, :, -shift_rc[0]:]
        kernel = F.pad(kernel, [0, 0, 0, -shift_rc[0]])
    if shift_rc[1] >= 0:
        kernel = F.pad(kernel, [shift_rc[1], 0, 0, 0])
        kernel = kernel[:, :, :, :col_ori]
    elif shift_rc[1] < 0:
        kernel = kernel[:, :, :, -shift_rc[1]:]
        kernel = F.pad(kernel, [0, -shift_rc[1], 0, 0])

    return kernel

def get_grid_rc(rc_size):
    row, col = rc_size
    row_mat=np.tile(np.arange(row)[:,np.newaxis],(1,col))
    col_mat=np.tile(np.arange(col)[np.newaxis,:],(row,1))
    return row_mat,col_mat

def get_tensor_centroid(tensor):
    assert tensor.dim()==3
    rc_size=tensor.shape[1:]
    row_mat,col_mat=get_grid_rc(rc_size)
    row_tensor=torch.from_numpy(row_mat.astype(np.float32)).unsqueeze(0)
    col_tensor=torch.from_numpy(col_mat.astype(np.float32)).unsqueeze(0)

    # 对输入进行归一化
    with torch.no_grad():
        tensor = tensor/torch.sum(tensor, dim=[1,2], keepdim=True)
        tensor = tensor.cpu()
        centroid_row=torch.sum(tensor*row_tensor, dim=[1,2]).numpy()
        centroid_col=torch.sum(tensor*col_tensor, dim=[1,2]).numpy()
    return centroid_row,centroid_col