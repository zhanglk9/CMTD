import cv2
import numpy as np
import colorsys
import torch

def id2rgb(id, max_num_obj=256):
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")

    # Convert the ID into a hue value
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)  # Ensure value is between 0 and 1
    s = 0.5 + (id % 2) * 0.5  # Alternate between 0.5 and 1.0
    l = 0.5

    # Use colorsys to convert HSL to RGB
    rgb = np.zeros((3,), dtype=np.uint8)
    if id == 0:  # invalid region
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r * 255), int(g * 255), int(b * 255)

    return rgb
def visualize_obj(objects):
    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        colored_mask = id2rgb(id)
        rgb_mask[objects == id] = colored_mask
    return rgb_mask

# 读取图片

# image = cv2.imread('3_00466_genid2664_labelTrainIds.png', cv2.IMREAD_GRAYSCALE)
# # import ipdb;ipdb.set_trace()
# image = visualize_obj(image.astype(np.uint8))
# torch.unique(torch.tensor(image))
#
#
# # 应用高斯模糊以减少噪声
# blurred = cv2.GaussianBlur(image, (5, 5), 0)
#
# # 使用 Canny 边缘检测
# edges = cv2.Canny(blurred, 30, 100)
# kernel = np.ones((3, 3), np.uint8)  # 3x3的膨胀核
# edges_thick = cv2.dilate(edges, kernel, iterations=1)
# # 显示结果
# cv2.imshow('Original Image', image)
# cv2.imshow('Edges', edges_thick)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



def color_transfer(source, reference):

    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2Lab)
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2Lab)

    l1, a1, b1 = cv2.split(source_lab)
    l2, a2, b2 = cv2.split(reference_lab)

    mean_src, std_src = [l1.mean(), a1.mean(), b1.mean()], [l1.std(), a1.std(), b1.std()]
    mean_ref, std_ref = [l2.mean(), a2.mean(), b2.mean()], [l2.std(), a2.std(), b2.std()]

    l = (l1 - mean_src[0]) * (std_ref[0] / std_src[0]) + mean_ref[0]
    a = (a1 - mean_src[1]) * (std_ref[1] / std_src[1]) + mean_ref[1]
    b = (b1 - mean_src[2]) * (std_ref[2] / std_src[2]) + mean_ref[2]

    transferred = cv2.merge([l.clip(0, 255).astype('uint8'),
                             a.clip(0, 255).astype('uint8'),
                             b.clip(0, 255).astype('uint8')])
    return cv2.cvtColor(transferred, cv2.COLOR_Lab2BGR)


def match_histograms(source, reference):

    source_channels = cv2.split(source)
    reference_channels = cv2.split(reference)

    matched_channels = []
    for sc, rc in zip(source_channels, reference_channels):

        src_hist, _ = np.histogram(sc, bins=256, range=(0, 256), density=True)
        ref_hist, _ = np.histogram(rc, bins=256, range=(0, 256), density=True)
        src_cdf = np.cumsum(src_hist)
        ref_cdf = np.cumsum(ref_hist)
        src_cdf = (src_cdf / src_cdf[-1]) * 255
        ref_cdf = (ref_cdf / ref_cdf[-1]) * 255
        mapping = np.interp(src_cdf, ref_cdf, np.arange(256))
        matched_channel = cv2.LUT(sc, mapping.astype(np.uint8))
        matched_channels.append(matched_channel)
    matched_image = cv2.merge(matched_channels)
    return matched_image



#存储新文件夹
import os
import shutil

ori_data_path = 'data/gta'
real_data_path = os.path.join('data', 'cityscapes','leftImg8bit','train','aachen')
new_data_path = 'data/basic_style'
if not os.path.exists(os.path.join(new_data_path,'images')):
    os.makedirs(os.path.join(new_data_path,'images'))
    os.makedirs(os.path.join(new_data_path,'labels'))

ori_data_folder = os.listdir(os.path.join(ori_data_path,'images'))
real_data_folder = os.listdir(real_data_path)
for i,frame in enumerate(ori_data_folder):
    if i % 6 == 0 :
        source_img = cv2.imread(os.path.join(ori_data_path,'images',frame))
        reference_img =cv2.imread(os.path.join(real_data_path,real_data_folder[i % len(real_data_folder)]))

        blurred = cv2.GaussianBlur(source_img, (3, 3), 0)
        color_matched_img = color_transfer(blurred, reference_img)
        matched_img = match_histograms(blurred, reference_img)

        output_path = os.path.join(new_data_path,'images', frame)
        cv2.imwrite(output_path, matched_img)
        new_file = frame.replace('.png', '_labelTrainIds.png')
        shutil.copy(os.path.join(ori_data_path,'labels',new_file), os.path.join(new_data_path,'labels', new_file))
        print(f"Saved {output_path}")

print("Complete transfer styles! ")


