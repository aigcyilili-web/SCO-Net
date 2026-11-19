import os
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from PIL import Image as Image
from data import TripleCompose, TripleRandomCrop, TripleRandomHorizontalFilp, \
    TripleRandomVerticalFlip, TripleToTensor
import torchvision.transforms as transforms
from torchvision.transforms import functional as trans_F
from torch.utils.data import Dataset, DataLoader
import random
import imgaug as ia
import math
from imgaug import augmenters as iaa
from fft_conv_utils import prepare_kernel, make_blurred, shift_kernel, \
    get_wn_fun, fft_conv_2d, pad_kernel
from cv_utils import enlarge_contrast

import os
from torch.utils.data import Dataset


ia.seed(random.randint(0, 100000))

# 注:当前版本输入网络的图像没有前置的提升亮度过程,输入网络的直接就是暗图像

def train_dataloader(path, batch_size=64, num_workers=0, use_transform=True,
                     psf_gene_obj=None,shuffle=True):
    image_dir = path

    transform = None
    if use_transform:
        transform = TripleCompose(
            [
                TripleRandomCrop(384),     #随机裁剪为384x384像素
                TripleRandomHorizontalFilp(),   #翻转
                TripleRandomVerticalFlip(),   #翻转
                TripleToTensor()  #tensor
            ]
        )
    dataloader = DataLoader(
        TrainDeblurDataset(image_dir, use_transform=use_transform, psf_gene_obj=psf_gene_obj),#psf_gene_obj=psf_gene_obj
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        # pin_memory=True
        pin_memory=False   #用true会快 回头试试
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path, 'test')
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DeblurDataset(os.path.join(path, 'valid')),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader


class TrainDeblurDataset(Dataset):
    def __init__(self, image_dir, use_transform=True, is_test=False, psf_gene_obj=None):
        self.image_dir = image_dir
        self.image_list = os.listdir(image_dir)  # 目录中的所有文件和子目录的名称
        self._check_image(self.image_list)
        self.image_list.sort()   #行字典序排序
        self.is_test = is_test   #
        self.psf_gene_obj=psf_gene_obj
        # 复现了利于反向传播的transform过程(主要是将flip操作提前)
        self.use_transform = use_transform
        self.crop_transform = TripleRandomCrop(384)

        # self.image_size = (960, 640)
        self.image_size = (960, 960)   #和测试集大小分开 正方形用于训练 960x640贴合实际图像
        # augment the image
        self.ill_seq_before, self.ill_seq_after = self.get_ill_seq()   #光照序列
        self.MAX_GRAY_VALUE = 255.
        # load the kernel
        if self.psf_gene_obj is None:
            self.kernel, self.kernel_size_half = prepare_kernel(self.image_size)   #对模糊核进行处理 ， 改变大小，填充，卷积等
        else:
            self.kernel=None
            self.kernel_size_half=None
        # light decreasing threshold
        self.low_V_ratio_min, self.low_V_ratio_max = 0.25, 0.30
        self.input_V_ratio_min, self.input_V_ratio_max = 0.75, 1.25    #光
        # parameter of wn filter
        self.wn_K_range = (0.004, 0.010)   #限制wn k的范围
        # 在边界附近掩模维纳滤波的结果
        self.corp_size = 384         #裁剪大小 和上面384一样
        wn_border_ratio_min = 0.05
        wn_border_ratio_max = 0.10
        self.full_border_r = (round(self.image_size[1] * wn_border_ratio_min),
                              round(self.image_size[1] * wn_border_ratio_max))
        self.full_border_c = (round(self.image_size[0] * wn_border_ratio_min),
                              round(self.image_size[0] * wn_border_ratio_max))
        self.crop_border_max = round(self.corp_size * wn_border_ratio_max)
        self.crop_mask_prob = 0.5   #在每次处理图像时，有50%的几率会应用掩码裁剪操作
        self.crop_mask_large_prob = 0.05   #  决定应用掩码裁剪的情况下，有5%的几率会使用较大的掩码区域
        # 设置不用于训练的样本数量
        # 现在的处理方式是:训练集中前out_valid_sample_num个样本不参与训练,用于验证
        self.out_valid_sample_num = 1000

    def __len__(self):
        return len(self.image_list)



    def __getitem__(self, idx):
        # # 调试阶段暂时屏蔽了该机制
        # # 训练集中前out_valid_sample_num个样本不参与训练,用于验证
        # # 所以当采样到这个范围的下标时,跳过并重新采样
        # if idx < self.out_valid_sample_num:
        #     idx = random.randint(self.out_valid_sample_num, len(self.image_list) - 1)

        # read the clear image
        image = cv2.imread(os.path.join(self.image_dir, self.image_list[idx]))
        image = cv2.resize(image, self.image_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        #输入数据增强（图像随机反转）
        if self.use_transform:
            # 随机镜像反转
            if random.random() < 0.5:
                # vertical flipping
                image=cv2.flip(image, flipCode=0)
            if random.random() < 0.5:
                # horizontal flipping
                image=cv2.flip(image, flipCode=1)

        # 随机选取亮度降低的比例
        low_V_ratio = ((self.low_V_ratio_max - self.low_V_ratio_min) * random.random() +
                       self.low_V_ratio_min)
        # 随机选取输入图像亮度降低的比例
        input_V_ratio = ((self.input_V_ratio_max - self.input_V_ratio_min) * random.random() +
                         self.input_V_ratio_min)
        rela_V_ratio = input_V_ratio / low_V_ratio

        # use_wn_mark用于标记是否使用维纳滤波
        use_wn_mark=False

        if self.psf_gene_obj is None:
            # 这种情况,是使用加工好的镜片进一步优化网络,因此对加载的固定psf做随机扩充
            # 随机微调成像模糊核
            kernel_thre_range = [0.01, 0.05]    #设置卷积核阈值范围，用于后续的阈值处理
            kernel_thre = np.random.uniform(*kernel_thre_range)  # 指定的阈值范围内随机选择一个阈值
            kernel = self.kernel.clone()   #创建原始卷积核的一个副本，避免修改原始核
            kernel_shift = shift_kernel(kernel, image_shift=(15, -6))   #  对卷积核进行平移操作，模拟图像偏移效果，物体或相机的运动有明确的方向性，传感器平面与镜头不完全平行，也可能产生轻微的不对称
            kernel_shift = torch.flip(kernel_shift, [2, 3])  # 对卷积核进行翻转操作，在维度2和3上进行翻转
            kernel_shift = kernel_shift[:, :, :self.image_size[1], :self.image_size[0]]   # 裁剪卷积核，使其不超过图像尺寸
            kernel_shift[kernel_shift < kernel_thre] = 0   #应用阈值处理：将卷积核中小于阈值的位置设置为0
            kernel_sum = torch.sum(torch.sum(kernel_shift, dim=3, keepdim=True), dim=2, keepdim=True)  #计算卷积核的总和，然后进行归一化
            kernel_shift = kernel_shift / kernel_sum
            # 抑制卷积核中很小的值
            kernel[kernel < kernel_thre] = 0  #对原始卷积核（未平移的）也进行相同的阈值处理和归一化
            kernel_sum = torch.sum(torch.sum(kernel, dim=3, keepdim=True), dim=2, keepdim=True)
            kernel = kernel / kernel_sum
            image_shift_value=(15, -3)
            kernel_size_half=self.kernel_size_half     #保存卷积核的一半尺寸，用于边界处理或卷积操作的计算
        else:
            # 这种情况,是仿真训练镜片参数的阶段,获取可学习的psf值
            # 点光源从xx米处入射第一块相位板
            # distance_range=(0.5, 5)
            distance_range=(1.5, 1.5)   #定义距离范围，这个距离很可能指的是物距（物体到镜头的距离）或像距（镜头到传感器的距离）
            distance = np.random.uniform(*distance_range)  #  从指定的距离范围内随机选择一个距离值
            point_light_shift_ratio=1.0   # 设置点光源偏移比例，用于计算最大偏移量
            point_light_shift_max=self.psf_gene_obj.wave_resolution[0]/2*point_light_shift_ratio #计算点光源的最大偏移量 可能表示波前分辨率或光学系统的分辨率
            point_light_shift=np.random.uniform(*(0,point_light_shift_max))   # 从0到最大偏移量之间随机选择一个偏移值 模拟了点光源在成像平面上的随机偏移
            shift_theta=np.random.uniform(*(0,2*math.pi))   # 随机生成一个角度值（0到2π之间）
            point_light_shift_xy=(point_light_shift*math.cos(shift_theta),   # 将极坐标下的偏移量转换为笛卡尔坐标系下的(x, y)偏移
                                  point_light_shift*math.sin(shift_theta))
            kernel=self.psf_gene_obj.get_psfs(distance, point_light_shift_xy,     #调用PSF生成对象的方法，获取点扩散函数（PSF）卷积核
                                              (self.image_size[1], self.image_size[0]),
                                              move_center_mark=False)###################################################True
            kernel,kernel_size_half=pad_kernel(kernel)   #对生成的PSF核进行填充操作
            image_shift_value=(0, 0)  #表示没有额外的图像偏移
            if use_wn_mark:
                # 在固定距离下获得psf值用于维纳滤波
                with torch.no_grad():
                    distance_fix = (distance_range[0]+distance_range[1])/2
                    point_light_shift_xy_fix=(0,0)
                    kernel_fix=self.psf_gene_obj.get_psfs(distance_fix, point_light_shift_xy_fix,
                                                        (self.image_size[1], self.image_size[0]))
                    kernel_shift_wn = torch.flip(kernel_fix.cpu(), [2, 3])
                    kernel_shift_wn = kernel_shift_wn[:, :, :self.image_size[1], :self.image_size[0]]

        # with torch.no_grad():
        #     temp=kernel[0].cpu().numpy().transpose((1,2,0))
        # cv2.imwrite('temp1.png',np.rint((temp-np.min(temp))/(np.max(temp)-np.min(temp))*255).astype('uint8'))

        # 得到仿真暗图像,以及后续输入图像应当对应的清晰图像(即网络学习的标签)
        #image_low, _ = self.change_image_light(
        #    image, ratio_list=[low_V_ratio, input_V_ratio])
        image_low = image#不进行变暗处理

        # label直接取原始图像
        image_label = image.copy()

        # 在仿真暗图像中加入高斯模糊
        #image_low = self.ill_seq_before.augment_image(image_low)
        # 在仿真暗图像中加入成像模糊
        image_tensor = torch.from_numpy(image_low[np.newaxis].transpose(0, 3, 1, 2)).float() / self.MAX_GRAY_VALUE   #将NumPy数组转换为PyTorch张量，并进行预处理：
        if self.psf_gene_obj is not None:    #检查是否存在PSF生成对象
            # 这种情况下,需要将图像数据加载到psf所在的device上
            image_tensor=image_tensor.to(kernel.device)
        image_new_list = []
        for channel_id in range(kernel.shape[1]):
            image_c_now = fft_conv_2d(image_tensor[:, channel_id:channel_id + 1],   #图像数据
                                      kernel[:, channel_id:channel_id + 1],  #通道的卷积核
                                      padding=kernel_size_half,  #通常为卷积核尺寸的一半
                                      image_shift=image_shift_value)   #   图像平移
            image_new_list.append(image_c_now.squeeze(0))
            # image_now = image_low[:, :, channel_id]
            # PSF = kernel_shift[0, channel_id].numpy()
            # image_now2 = np.abs(make_blurred(image_now, PSF, 1e-3)) #模糊被#
            # image_new_list.append(image_now2[:, :, np.newaxis])

        image_low_tensor = torch.cat(image_new_list,dim=0) * self.MAX_GRAY_VALUE
        image_low_np_clean = image_low_tensor.detach().cpu().numpy().transpose(1, 2, 0)
        image_low_np_clean = np.rint(image_low_np_clean).astype('uint8')
        # 在仿真暗图像中加入噪声(必须是先模糊再加入噪声)
        image_low_np = self.ill_seq_after.augment_image(image_low_np_clean)      #加入噪声
        # 将numpy array中加入的噪声叠加在tensor上
        add_noise=image_low_np.astype(np.float32)-image_low_np_clean.astype(np.float32)
        add_noise=torch.from_numpy(add_noise.transpose(2,0,1)).to(image_low_tensor.device)
        image_low_tensor=image_low_tensor+add_noise

        # 得到图像的维纳滤波结果
        if use_wn_mark:
            K_value = self.wn_K_range[0] + (self.wn_K_range[1] - self.wn_K_range[0]) * random.random()
            image_low_wn_np = get_wn_fun(iemag_low_np, kernel_shift_wn, K_value)

            # 随机采样边界掩模尺寸
            mask_r1 = random.randint(self.full_border_r[0], self.full_border_r[1])  #随机生成上边界的掩码宽度
            mask_r2 = random.randint(self.full_border_r[0], self.full_border_r[1])
            mask_c1 = random.randint(self.full_border_c[0], self.full_border_c[1])
            mask_c2 = random.randint(self.full_border_c[0], self.full_border_c[1])
            # 对全图维纳滤波结果的边界区域进行随机掩模
            image_low_wn_np[:mask_r1] = 0  #遮挡图像的上边界区域
            image_low_wn_np[self.image_size[1] - mask_r2:] = 0
            image_low_wn_np[:, :mask_c1] = 0
            image_low_wn_np[:, self.image_size[0] - mask_c2:] = 0
           #通过随机遮挡边界区域，增加训练数据的多样性
           #防止模型过度依赖图像边界信息进行预测
           #扫描件边缘阴影、相机镜头遮挡等
        # cv2.imwrite('debug_image/result_1.bmp', cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR))
        # cv2.imwrite('debug_image/result_2.bmp', cv2.cvtColor(image_input_wn, cv2.COLOR_RGB2BGR))
        # cv2.imwrite('debug_image/result_3.bmp', cv2.cvtColor(image_label, cv2.COLOR_RGB2BGR))

        image_input=image_low_tensor
        if use_wn_mark:
            image_input_wn=image_low_wn_np
        else:
            image_input_wn=np.zeros_like(image_low_np)
        if self.psf_gene_obj is not None:
            # image_input已经是tensor格式,无需数据类型转换
            # 除image_input外另外两个数据均为numpy的array,需转为PIL image再处理
            image_input_wn = Image.fromarray(image_input_wn)
            image_label = Image.fromarray(image_label)

            # 根据需要随机裁剪图像  #此处随机裁剪是考虑显存问题，不是必须裁剪的
            if self.use_transform:
                image_input, image_input_wn, image_label = self.crop_transform(
                        image_input, image_input_wn, image_label)

            # 归一化,统一通道次序
            assert self.MAX_GRAY_VALUE==255.
            image_input=image_input/self.MAX_GRAY_VALUE
            image_input_wn = trans_F.to_tensor(image_input_wn)
            image_label = trans_F.to_tensor(image_label)

            # 对crop后的维纳滤波结果再做一次掩模,尝试提升泛化能力
            crop_rand_value = random.random()
            if use_wn_mark and self.use_transform and (crop_rand_value < self.crop_mask_prob):  #掩码标记    数据变换  随机值是否小于掩码裁剪概率
                if crop_rand_value < self.crop_mask_large_prob:   #  嵌套条件判断，检查随机值是否小于大掩码概率
                    mask_vec = np.array(((mask_r1, mask_r2), (mask_c1, mask_c2)), dtype='int')  #创建一个2×2的NumPy数组，包含行和列的边界信息
                    zero_mask_pos = np.random.choice(2, size=(2,), replace=True)  #  随机选择两个位置（每个位置从0或1中选择）
                    mask_vec[np.arange(2), zero_mask_pos] = 0  #  将mask_vec数组中特定位置的值设置为0
                    mask_crop_r1 = int(mask_vec[0, 0])
                    mask_crop_r2 = int(mask_vec[0, 1])
                    mask_crop_c1 = int(mask_vec[1, 0])  # 从修改后的mask_vec中提取行和列的边界值
                    mask_crop_c2 = int(mask_vec[1, 1])
                else:
                    mask_crop_r1 = random.randint(0, self.crop_border_max)
                    mask_crop_r2 = random.randint(0, self.crop_border_max)
                    mask_crop_c1 = random.randint(0, self.crop_border_max)
                    mask_crop_c2 = random.randint(0, self.crop_border_max)
                image_input_wn[:, :mask_crop_r1] = 0
                image_input_wn[:, self.corp_size - mask_crop_r2:] = 0
                image_input_wn[:, :, :mask_crop_c1] = 0
                image_input_wn[:, :, self.corp_size - mask_crop_c2:] = 0

            # 这种情况下,需要将维纳滤波结果加载到psf所在的device上
            image_input_wn=image_input_wn.to(image_input.device)
            image_input = torch.cat((image_input, image_input_wn), dim=0)
            distance_tensor = torch.tensor([distance,], dtype=image_input.dtype,
                                           device=image_input.device)
        else:
            image_input = Image.fromarray(image_input)
            image_input_wn = Image.fromarray(image_input_wn)
            image_label = Image.fromarray(image_label)
            image_input = trans_F.to_tensor(image_input)
            image_input_wn = trans_F.to_tensor(image_input_wn)
            image_label = trans_F.to_tensor(image_label)
            image_input = torch.cat((image_input, image_input_wn), dim=0)
            distance_tensor = torch.zeros((1,), dtype=image_input.dtype,
                                          device=image_input.device)
        if self.is_test:
            name = self.image_list[idx]
            return image_input, image_label, name

        # with torch.no_grad():
        #     image_np = np.rint(image_input[:3].cpu().numpy().transpose((1, 2, 0)) * 255).astype('uint8')
        #     image2_np = np.rint(image_input[3:].cpu().numpy().transpose((1, 2, 0)) * 255).astype('uint8')
        #     label_np = np.rint(image_label.cpu().numpy().transpose((1, 2, 0)) * 255).astype('uint8')
        #     cv2.imwrite('debug_image/result_1.bmp', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        #     cv2.imwrite('debug_image/result_2.bmp', cv2.cvtColor(image2_np, cv2.COLOR_RGB2BGR))
        #     cv2.imwrite('debug_image/result_3.bmp', cv2.cvtColor(label_np, cv2.COLOR_RGB2BGR))
        #     kernel_np=kernel[0].cpu().numpy().transpose((1,2,0))
        #     kernel_np=np.rint((kernel_np-np.min(kernel_np))/(np.max(kernel_np)-np.min(kernel_np))*255).astype('uint8')
        #     cv2.imwrite('debug_image/kernel.bmp',cv2.cvtColor(kernel_np, cv2.COLOR_RGB2BGR))
        # image_label = cv2.imread('debug_image/%d_1.jpg' % idx)
        # image_label = cv2.cvtColor(image_label, cv2.COLOR_BGR2RGB)
        # image_input = cv2.imread('debug_image/%d_2.jpg' % idx)
        # image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        # image_input = trans_F.to_tensor(image_input)
        # image_label = trans_F.to_tensor(image_label)

        return image_input, image_label, kernel.squeeze(0), distance_tensor

    def get_ill_seq(self):
        seq_before = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=1)),
        ])

        seq_after = iaa.Sequential([
            iaa.AdditiveGaussianNoise(0, (1, 2)),
            iaa.AdditivePoissonNoise((1, 2)),
            iaa.Sometimes(0.9, iaa.JpegCompression((5, 20))),
        ])

        return seq_before, seq_after

    def change_image_light(self, image, ratio=None, ratio_list=None):
        assert (ratio is not None) or (ratio_list is not None)

        if ratio is not None:
            ratio_list = [ratio]

        ratio_num = len(ratio_list)
        image_new_list = []
        for image_id in range(ratio_num):
            # image_hsv = cv2.cvtColor(image.astype('float32'), cv2.COLOR_RGB2HSV_FULL)
            # image_low_hsv = image_hsv.copy()
            # image_low_hsv[:, :, 2] = image_hsv[:, :, 2] * ratio_list[image_id]
            # image_new = cv2.cvtColor(image_low_hsv, cv2.COLOR_HSV2RGB_FULL)
            image_new = image.astype('float32') * ratio_list[image_id]

            image_new = np.maximum(0, np.minimum(255, image_new))
            image_new = np.rint(image_new).astype('uint8')
            image_new_list.append(image_new)

        if ratio is not None:
            image_new = image_new_list[0]
        else:
            image_new = image_new_list

        return image_new



####   666      图片格式


    @staticmethod
    def _check_image(lst):
        # 允许的文件格式列表，添加svs和SVS
        allowed_formats = ['png', 'jpg', 'jpeg', 'JPG', 'bmp', 'svs', 'SVS']
        for x in lst:
            splits = x.split('.')
            # 确保文件有后缀
            if len(splits) < 2:
                raise ValueError(f"文件 {x} 没有后缀名，不支持的格式")
            if splits[-1] not in allowed_formats:
                raise ValueError(f"文件 {x} 的格式不支持，仅支持 {allowed_formats}")

    # def _check_image(lst):
    #     for image_path in lst:
    #         if image_path.endswith('.svs'):
    #             try:
    #                 slide = OpenSlide(image_path)
    #                 # 简单验证文件可打开
    #                 slide.close()
    #             except:
    #                 raise ValueError(f"Invalid .svs file: {image_path}")
    #         else:
    #             try:
    #                 with Image.open(image_path) as img:
    #                     img.verify()  # 验证图像完整性
    #             except:
    #                 raise ValueError(f"Invalid image file: {image_path}")


class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'valid/'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

        # load the kernel
        self.MAX_GRAY_VALUE = 255.
        self.image_size = (960, 640)
  #      extra_ratio = (50, 50)  #裁剪？
   #     self.image_size = tuple(round(item * extra_ratio[k]) for k, item in enumerate(self.image_size))
        self.kernel, self.kernel_size_half = prepare_kernel(self.image_size)
        # 固定的成像模糊核,用于维纳滤波
        kernel_thre_range = [0.01, 0.05]
        kernel_thre = (kernel_thre_range[0] + kernel_thre_range[1]) / 2
        kernel_shift = shift_kernel(self.kernel, image_shift=(15, -6))
        kernel_shift = torch.flip(kernel_shift, [2, 3])
        kernel_shift = kernel_shift[:, :, :self.image_size[1], :self.image_size[0]]
        kernel_shift[kernel_shift < kernel_thre] = 0
        kernel_sum = torch.sum(torch.sum(kernel_shift, dim=3, keepdim=True), dim=2, keepdim=True)
        self.kernel_shift = kernel_shift / kernel_sum
        # parameter of wn filter
        self.wn_K_range = (0.004, 0.010)
        self.K_value = (self.wn_K_range[0] + self.wn_K_range[1]) / 2
        # 在边界附近掩模维纳滤波的结果
        corp_size = 384
        wn_border_ratio_min = 0.05
        wn_border_ratio_max = 0.10
        self.wn_border_ratio = 0.08

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_ori = cv2.imread(os.path.join(self.image_dir, 'blur', self.image_list[idx]))
        image_ori = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
        label = cv2.imread(os.path.join(self.image_dir, 'sharp', self.image_list[idx]))
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        # # 线性校正
        # image, enlarge_ratio = enlarge_contrast(image_ori)
        # 消融实验,图像在输入网络前不做亮度变化
        image = image_ori

        # 得到维纳滤波结果
        # image_input_wn = get_wn_fun(image, self.kernel_shift, self.K_value)
        # 消融实验,不加入维纳滤波的结果
        image_input_wn = np.zeros_like(image)

        # 对全图维纳滤波结果的边界区域进行掩模
        image_rows, image_cols = image.shape[:2]
        mask_r = round(image_rows * self.wn_border_ratio)
        mask_c = round(image_cols * self.wn_border_ratio)
        image_input_wn[:mask_r] = 0
        image_input_wn[image_rows - mask_r:] = 0
        image_input_wn[:, :mask_c] = 0
        image_input_wn[:, image_cols - mask_c:] = 0

        # cv2.imwrite('debug_image/result_1.bmp', cv2.cvtColor(image_ori, cv2.COLOR_RGB2BGR))
        # cv2.imwrite('debug_image/result_2.bmp', cv2.cvtColor(image_input_wn, cv2.COLOR_RGB2BGR))
        # cv2.imwrite('debug_image/result_3.bmp', cv2.cvtColor(label, cv2.COLOR_RGB2BGR))

        image = trans_F.to_tensor(image)
        image_input_wn = trans_F.to_tensor(image_input_wn)
        label = trans_F.to_tensor(label)

        image = torch.cat((image, image_input_wn), dim=0)
        if self.is_test:
            name = self.image_list[idx]
            return image, label, name
        return image, label

    @staticmethod
    def _check_image(lst):
        # 允许的文件格式列表，添加svs和SVS
        allowed_formats = ['png', 'jpg', 'jpeg', 'JPG', 'bmp', 'svs', 'SVS']
        for x in lst:
            splits = x.split('.')
            # 确保文件有后缀
            if len(splits) < 2:
                raise ValueError(f"文件 {x} 没有后缀名，不支持的格式")
            if splits[-1] not in allowed_formats:
                raise ValueError(f"文件 {x} 的格式不支持，仅支持 {allowed_formats}")
    # def _check_image(lst):
    #     for image_path in lst:
    #         if image_path.endswith('.svs'):
    #             try:
    #                 slide = OpenSlide(image_path)
    #                 # 简单验证文件可打开
    #                 slide.close()
    #             except:
    #                 raise ValueError(f"Invalid .svs file: {image_path}")
    #         else:
    #             try:
    #                 with Image.open(image_path) as img:
    #                     img.verify()  # 验证图像完整性
    #             except:
    #                 raise ValueError(f"Invalid image file: {image_path}")