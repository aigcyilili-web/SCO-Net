import math
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lens_utils
import matplotlib.pyplot as plt
from .layers import *


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class SCM(nn.Module):
    def __init__(self, input_plane, image_channels, out_plane):
        super(SCM, self).__init__()

        self.image_channels = image_channels

        self.main = nn.Sequential(
            BasicConv(input_plane, out_plane // 4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane - image_channels, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        # 注意x中只有原始图像的image_channel个通道参与堆叠,后续辅助通道不参与堆叠
        x = torch.cat([x[:, :self.image_channels], self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out


class MIMOUNet(nn.Module):
    def __init__(self, num_res=8):
        super(MIMOUNet, self).__init__()

        self.input_plane = 6
        self.image_channels = 3
        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel * 2, num_res),
            EBlock(base_channel * 4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(self.input_plane, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, self.image_channels, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, self.image_channels, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, self.image_channels, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel * 1),
            AFF(base_channel * 7, base_channel * 2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(self.input_plane, self.image_channels, base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(self.input_plane, self.image_channels, base_channel * 2)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5, recompute_scale_factor=False)
        x_4 = F.interpolate(x_2, scale_factor=0.5, recompute_scale_factor=False)
        image_0 = x[:, :self.image_channels]
        image_2 = x_2[:, :self.image_channels]
        image_4 = x_4[:, :self.image_channels]

        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5, recompute_scale_factor=False)
        z21 = F.interpolate(res2, scale_factor=2, recompute_scale_factor=False)
        z42 = F.interpolate(z, scale_factor=2, recompute_scale_factor=False)
        z41 = F.interpolate(z42, scale_factor=2, recompute_scale_factor=False)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_ + image_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_ + image_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z + image_0)

        return outputs

class LensModule(nn.Module):
    def __init__(self):
        super(LensModule, self).__init__()

        #################### 透镜相关 #####################
        self.wave_resolution = (1872, 1872)
        self.input_sample_interval = 3e-6
        #单位是m,故左式代表640,550,460nm
        self.wave_lengths = torch.tensor(np.array([640, 550, 460]) * 1e-9, dtype=torch.float32)
        #AZ1518光刻胶折射率
        self.refractive_idcs = np.array([1.6983, 1.7091, 1.7302])
        assert self.wave_resolution[0]==self.wave_resolution[1]
        
        # 可加工的透镜高度范围
        self.max_height=2e-6
        self.min_height=0
        # self.lens_layer1 = nn.Parameter(torch.full(self.wave_resolution, fill_value=-2, 
        #                                            dtype=torch.float32))
        # self.lens_layer2 = nn.Parameter(torch.full(self.wave_resolution, fill_value=-2, 
        #                                            dtype=torch.float32))


        self.lens_layer1 = nn.Parameter(torch.zeros(self.wave_resolution, dtype=torch.float32))#屏蔽随机初始分布
        #第一层使用余弦初始分布
        # self.init_height_cos = lens_utils.get_init_cos_height(self.wave_resolution)
        # arctanh_init_height_cos = torch.arctanh(self.init_height_cos / (self.max_height / 2))
        # self.lens_layer1 = nn.Parameter(arctanh_init_height_cos.float())

        # 镜片2加入一个初始高度init_height，作为训练初始条件，理论上更利于搜索极小值
        self.init_height = lens_utils.get_init_height(self.wave_resolution)
        arctanh_init_height = torch.arctanh(self.init_height/(self.max_height/2))
        self.lens_layer2 = nn.Parameter(arctanh_init_height.float())
        temp=1
        
    def get_usable_height(self, max_noise_ratio):
        # height_1=self.min_height+torch.sigmoid(self.lens_layer1)*(self.max_height-self.min_height)
        # height_2=self.min_height+torch.sigmoid(self.lens_layer2)*(self.max_height-self.min_height)
        # height_2=height_2 + self.init_height.to(self.lens_layer2.device)
        height_1=torch.tanh(self.lens_layer1)*self.max_height/2
        height_2=torch.tanh(self.lens_layer2)*self.max_height/2
        # with torch.no_grad():
        #     height_1_min=torch.min(height_1)
        #     height_2_min=torch.min(height_2)
        height_1_min=torch.min(height_1)
        height_2_min=torch.min(height_2)
        height_1=height_1-height_1_min
        height_2=height_2-height_2_min
        
        if max_noise_ratio<(1e-16):
            return height_1,height_2
        with torch.no_grad():
            height_1_noise_ratio=torch.rand_like(height_1)*(2*max_noise_ratio)-max_noise_ratio
            height_2_noise_ratio=torch.rand_like(height_2)*(2*max_noise_ratio)-max_noise_ratio
            height_1_new=height_1*(1+height_1_noise_ratio)
            height_2_new=height_2*(1+height_2_noise_ratio)
            # 截断到可加工的高度范围
            height_1_new[height_1_new<self.min_height]=self.min_height
            height_1_new[height_1_new>self.max_height]=self.max_height
            height_2_new[height_2_new<self.min_height]=self.min_height
            height_2_new[height_2_new>self.max_height]=self.max_height
            height1_change=height_1_new-height_1
            height2_change=height_2_new-height_2
        height_1=height_1+height1_change
        height_2=height_2+height2_change
        return height_1,height_2
       
    def height_out_loss(self):
        # 得到松弛后的高度边界
        out_ratio=0.1
        self.max_height=2e-6
        min_height=self.min_height-(self.max_height-self.min_height)*out_ratio
        max_height=self.max_height+(self.max_height-self.min_height)*out_ratio
        # 得到超出边界的高度的损失
        height_1_ori, height_2_ori = self.get_usable_height(max_noise_ratio=0)
        with torch.no_grad():
            small_1_mask=(height_1_ori<min_height)
            small_1_num=torch.sum(small_1_mask).item()
            large_1_mask=(height_1_ori>max_height)
            large_1_num=torch.sum(large_1_mask).item()
            small_2_mask=(height_2_ori<min_height)
            small_2_num=torch.sum(small_2_mask).item()
            large_2_mask=(height_2_ori>max_height)
            large_2_num=torch.sum(large_2_mask).item()
        loss=torch.tensor(0,dtype=height_1_ori.dtype,device=height_1_ori.device)
        if small_1_num>0:
            loss=torch.sum(min_height-height_1_ori[small_1_mask])+loss
        if large_1_num>0:
            loss=torch.sum(height_1_ori[large_1_mask]-max_height)+loss
        if small_2_num>0:
            loss=torch.sum(min_height-height_2_ori[small_2_mask])+loss
        if large_2_num>0:
            loss=torch.sum(height_2_ori[large_2_mask]-max_height)+loss
            
        all_out_num=small_1_num+large_1_num+small_2_num+large_2_num
        if all_out_num>0:
            loss=loss/all_out_num
            
        return loss
    
    def height_smooth_loss(self):
        rad_sample_num=100
        rad_min=20
        rad_max=self.wave_resolution[0]/2
        height_diff_large_num=10
        
        height_1, height_2 = self.get_usable_height(max_noise_ratio=0)
        height_list=[height_1.unsqueeze(0).unsqueeze(0),height_2.unsqueeze(0).unsqueeze(0)]
        center_x=(self.wave_resolution[1]-1)/2
        center_y=(self.wave_resolution[0]-1)/2
        
        loss=0
        for height_id in range(len(height_list)):
            height_tensor=height_list[height_id]
            rad_vec=rad_min+np.random.random((rad_sample_num,))*(rad_max-rad_min)
            for rad_id in range(rad_sample_num):
                rad_now=rad_vec[rad_id]
                perimeter=round(2*rad_now*math.pi)
                theta=np.arange(perimeter)/perimeter*(2*math.pi)
                x=rad_now*np.cos(theta)+center_x
                y=rad_now*np.sin(theta)+center_y
                x=x/(self.wave_resolution[1])*2-1
                y=y/(self.wave_resolution[0])*2-1
                xy=np.concatenate((x[:,np.newaxis],y[:,np.newaxis]),axis=1)
                xy=torch.from_numpy(xy.astype(np.float32)).view(1,1,perimeter,2)
                xy=xy.to(height_1.device)
                height_circle=F.grid_sample(height_tensor, xy, align_corners=False)
                height_circle=height_circle.squeeze()
                
                # 计算标准差作为损失
                loss=loss+torch.std(height_circle)
                
                # # 只计算偏差大的位置的损失
                # height_mean=torch.mean(height_circle)
                # height_diff=torch.abs(height_circle-height_mean)
                # height_diff_large,_=torch.topk(height_diff,height_diff_large_num)
                # loss=loss+torch.sum(height_diff_large)
                
                # # 检验sample出的位置是否正确
                # row_mat,col_mat=lens_utils.get_grid_rc(self.wave_resolution)
                # rc_tensor=np.concatenate((row_mat[np.newaxis],col_mat[np.newaxis]),axis=0)
                # rc_tensor=torch.from_numpy(rc_tensor.astype(np.float32)).unsqueeze(0)
                # rc_sample=F.grid_sample(rc_tensor, xy.cpu(), align_corners=False)
                # rc_sample=np.rint(rc_sample.numpy()).astype(np.int64)
                # rc_sample=rc_sample.squeeze(2).squeeze(0).transpose((1,0))
                # mask=np.zeros(self.wave_resolution,dtype=np.uint8)
                # mask[rc_sample[:,0],rc_sample[:,1]]=255
                # import cv2
                # cv2.imwrite('debug_image/mask.png',mask)
                temp=1
                
        loss=loss/(rad_sample_num*len(height_list))
        return loss
                
    def get_lens_phase(self, height_1, height_2):
        device = self.lens_layer1.device
        
        delta_N = torch.tensor(self.refractive_idcs.reshape([1, 1, 1, -1]) - 1.0, dtype=torch.float32)
        wave_nos = 2. * np.pi / self.wave_lengths
        wave_nos = wave_nos.reshape([1, 1, 1, -1])
    
        ###通过高度分布，计算出layer1第一块相位板的相位
        phase1 =  (wave_nos * delta_N).to(device) * height_1.unsqueeze(-1).unsqueeze(0)
        phase1 =  phase1.to(torch.complex64)
        phase1 =  torch.exp(1j * phase1)
        ###通过高度分布，计算出layer2第二块相位板的相位
        phase2 =  (wave_nos * delta_N).to(device) * (height_2.unsqueeze(-1).unsqueeze(0))
        phase2 =  phase2.to(torch.complex64)
        phase2 =  torch.exp(1j * phase2)
        
        return phase1, phase2
        
    def get_psfs(self, distance, point_light_shift_xy, kernel_size, move_center_mark=False):
        physical_size = self.wave_resolution[0] * self.input_sample_interval
        phaplate_distance = torch.tensor(15e-3, dtype=torch.float32) + torch.rand((), dtype=torch.float32) * 0.0002 - 0.0001###########15e-3
        sensor_distance = torch.tensor(27e-3, dtype=torch.float32) + torch.rand((), dtype=torch.float32) * 0.0002 - 0.0001########32e-3
        
        # 得到加入噪声并截断至可用范围的高度
        max_noise_ratio=0.05
        height_1_usable, height_2_usable=self.get_usable_height(max_noise_ratio)
        
        phase1, phase2 = self.get_lens_phase(height_1_usable, height_2_usable)
        psf = lens_utils.get_psfs(
            distance, sensor_distance, phaplate_distance, self.wave_resolution, physical_size, 
            self.wave_lengths, phase1, phase2, 
            self.input_sample_interval, kernel_size, point_light_shift_xy)
    
        if move_center_mark:
            ############### 纠正psf中心位置 ######################
            # # 人为将psf的最大值移动至中心位置
            # with torch.no_grad():
            #     max_pos=torch.argmax(torch.sum(psf,dim=1)).item()
            # max_row=math.floor(max_pos/kernel_size[1])
            # max_col=max_pos%kernel_size[1]
        
            # 人为将psf的质心移动至中心位置
            with torch.no_grad():
                psf_sum=torch.sum(psf,dim=1)
                max_row,max_col=lens_utils.get_tensor_centroid(psf_sum)
            assert psf_sum.shape[0]==1
            max_row=max_row[0]
            max_col=max_col[0]
            
            center_row=(kernel_size[0]-1)/2
            center_col=(kernel_size[1]-1)/2
            shift_row=round(center_row-max_row)
            shift_col=round(center_col-max_col)
            psf=lens_utils.shift_tensor(psf,(shift_row,shift_col))
        
        return psf
    
    def save_lens_fig(self, dir_path):
        # 保存镜片高度和相位
        with torch.no_grad():
            height_1, height_2 = self.get_usable_height(max_noise_ratio=0)
            phase_1, phase_2 = self.get_lens_phase(height_1,height_2)
            angle_1 = torch.angle(phase_1)
            angle_2 = torch.angle(phase_2)
            height_1=height_1.cpu().numpy()
            height_2=height_2.cpu().numpy()
            plt.imshow(height_1)
            plt.colorbar()
            plt.savefig(osp.join(dir_path, 'height1.png'))
            plt.clf()
            plt.imshow(height_2)
            plt.colorbar()
            plt.savefig(osp.join(dir_path, 'height2.png'))
            plt.clf()
            
            angle_1=angle_1.cpu().numpy()
            angle_2=angle_2.cpu().numpy()
            for channel_id in range(3):#导出两个镜片对应三个波长通道的相位分布
                plt.imshow(angle_1[0,:,:,channel_id])
                plt.colorbar()
                plt.savefig(osp.join(dir_path, 'angle1_%d.png'%channel_id))
                plt.clf()
                plt.imshow(angle_2[0,:,:,channel_id])
                plt.colorbar()
                plt.savefig(osp.join(dir_path, 'angle2_%d.png'%channel_id))
                plt.close()
            plt.close()

def build_net(model_name):
    class ModelError(Exception):
        def __init__(self, msg):
            self.msg = msg

        def __str__(self):
            return self.msg

    if model_name == "MIMO-UNet":
        return MIMOUNet()
    raise ModelError('Wrong Model!\nYou should choose MIMO-UNetPlus or MIMO-UNet.')
