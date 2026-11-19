import numpy as np
import cv2
import math
import matplotlib.pyplot as plt


def change_image_light(image, ratio=None, ratio_list=None):
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


def enlarge_contrast(image, saturate_ratio=0.000):
    # saturate_ratio=0.005
    MAX_GRAY_VALUE = 255.0

    image_vec = image.reshape(-1)
    value_num = image_vec.shape[0]
    saturate_num = round((value_num - 1) * (1 - saturate_ratio))
    saturate_ori_value = np.partition(image_vec, saturate_num)[saturate_num]
    saturate_ori_value = min(saturate_ori_value + 1, MAX_GRAY_VALUE)
    ratio = MAX_GRAY_VALUE / saturate_ori_value

    image_new = image.astype('float') * ratio
    image_new = np.maximum(0, np.minimum(MAX_GRAY_VALUE, image_new))
    image_new = np.rint(image_new).astype('uint8')
    return image_new, ratio


def change_image_light_RGB(image, ratio):
    MAX_GRAY_VALUE = 255.0

    image_new = image.astype('float') * ratio
    image_new = np.maximum(0, np.minimum(MAX_GRAY_VALUE, image_new))
    image_new = np.rint(image_new).astype('uint8')
    return image_new


def enlarge_contrast_gamma(image):
    MAX_GRAY_VALUE = 255.0
    CENTER_GRAY_NORM = 0.5

    image_hsv = cv2.cvtColor(image.astype('float32'), cv2.COLOR_RGB2HSV_FULL)

    # 统计得到较好的gamma值
    V_mat = image_hsv[:, :, 2] / MAX_GRAY_VALUE
    # V_center = np.median(V_mat)
    V_center = (np.max(V_mat) + np.min(V_mat)) / 2
    if V_center > CENTER_GRAY_NORM:
        return image, -1
    gamma = math.log(CENTER_GRAY_NORM, V_center)
    assert gamma <= 1.0000000000001

    # # 统计得到较好的gamma值,尝试2
    # V_mat = image_hsv[:, :, 2] / MAX_GRAY_VALUE
    # V_center = np.median(V_mat)
    # CENTER_GRAY_NORM = (np.max(V_mat) + np.min(V_mat)) / 2
    # gamma = math.log(CENTER_GRAY_NORM, V_center)

    # 利用gamma调整亮度值
    V_mat_new = np.power(V_mat, gamma)
    image_hsv[:, :, 2] = V_mat_new * MAX_GRAY_VALUE
    image_new = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB_FULL)
    image_new = np.maximum(0, np.minimum(MAX_GRAY_VALUE, image_new))
    image_new = np.rint(image_new).astype('uint8')

    return image_new, gamma


def change_image_light_gamma(image, gamma):
    MAX_GRAY_VALUE = 255.0

    image_hsv = cv2.cvtColor(image.astype('float32'), cv2.COLOR_RGB2HSV_FULL)
    V_mat = image_hsv[:, :, 2] / MAX_GRAY_VALUE
    # 利用gamma调整亮度值
    V_mat_new = np.power(V_mat, gamma)
    image_hsv[:, :, 2] = V_mat_new * MAX_GRAY_VALUE
    image_new = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB_FULL)
    image_new = np.maximum(0, np.minimum(MAX_GRAY_VALUE, image_new))
    image_new = np.rint(image_new).astype('uint8')
    return image_new


def equalizeHist_hsv(image):
    image_hsv = cv2.cvtColor(image.astype('float32'), cv2.COLOR_RGB2HSV_FULL)

    V_mat = np.rint(image_hsv[:, :, 2]).astype('uint8')
    V_mat_new = cv2.equalizeHist(V_mat)

    image_hsv[:, :, 2] = V_mat_new
    image_new = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB_FULL)
    image_new = np.rint(image_new).astype('uint8')

    return image_new
