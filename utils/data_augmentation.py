import math
import cv2
import numpy
from PIL import ImageEnhance
import os
import numpy as np
from PIL import Image


def label_without_variety(root_path, img_name):
    image = Image.open(os.path.join(root_path, img_name))
    return image


def brightnessEnhancement(root_path, img_name):  # 亮度增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_bri = ImageEnhance.Brightness(image)
    # brightness = 1.1+0.4*np.random.random()#取值范围1.1-1.5
    brightness = 1.5
    image_brightened = enh_bri.enhance(brightness)
    return image_brightened


def contrastEnhancement(root_path, img_name):  # 对比度增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_con = ImageEnhance.Contrast(image)
    # contrast = 1.1+0.4*np.random.random()#取值范围1.1-1.5
    contrast = 1.5
    image_contrasted = enh_con.enhance(contrast)
    return image_contrasted


def rotation(root_path, img_name, boundary_path, mask_path):
    img = Image.open(os.path.join(root_path, img_name))
    boundary = Image.open(os.path.join(boundary_path, img_name))
    mask = Image.open(os.path.join(mask_path, img_name))

    random_angle = np.random.randint(-2, 2) * 90
    if random_angle == 0:
        rotation_img = img.rotate(-90)  # 旋转角度
        rotation_boundary = boundary.rotate(-90)
        rotation_mask = mask.rotate(-90)
    else:
        rotation_img = img.rotate(random_angle, expand=True)  # 旋转角度
        rotation_boundary = boundary.rotate(random_angle, expand=True)
        rotation_mask = mask.rotate(random_angle, expand=True)

    return rotation_img, rotation_boundary, rotation_mask


def flip(root_path, img_name, boundary_path, mask_path):  # 翻转图像
    img = Image.open(os.path.join(root_path, img_name))
    boundary = Image.open(os.path.join(boundary_path, img_name))
    mask = Image.open(os.path.join(mask_path, img_name))

    filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    filp_boundary = boundary.transpose(Image.FLIP_LEFT_RIGHT)
    filp_mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    return filp_img, filp_boundary, filp_mask


def fangshe_bianhuan(root_path, img_name, boundary_path, mask_path):  # 仿射变化扩充图像
    img = Image.open(os.path.join(root_path, img_name))
    boundary = Image.open(os.path.join(boundary_path, img_name))
    mask = Image.open(os.path.join(mask_path, img_name))

    img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
    boundary = cv2.cvtColor(numpy.asarray(boundary), cv2.COLOR_RGB2BGR)
    mask = cv2.cvtColor(numpy.asarray(mask), cv2.COLOR_RGB2BGR)

    h, w = img.shape[0], img.shape[1]
    m = cv2.getRotationMatrix2D(center=(w // 2, h // 2), angle=-30, scale=0.5)

    r_img = cv2.warpAffine(src=img, M=m, dsize=(w, h), borderValue=(0, 0, 0))
    r_boundary = cv2.warpAffine(src=boundary, M=m, dsize=(w, h), borderValue=(0, 0, 0))
    r_mask = cv2.warpAffine(src=mask, M=m, dsize=(w, h), borderValue=(0, 0, 0))

    r_img = Image.fromarray(cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB))
    r_boundary = Image.fromarray(cv2.cvtColor(r_boundary, cv2.COLOR_BGR2RGB))
    r_mask = Image.fromarray(cv2.cvtColor(r_mask, cv2.COLOR_BGR2RGB))
    return r_img, r_boundary, r_mask


def cuoqie(root_path, img_name, boundary_path, mask_path):  # 错切变化扩充图像
    img = Image.open(os.path.join(root_path, img_name))
    boundary = Image.open(os.path.join(boundary_path, img_name))
    mask = Image.open(os.path.join(mask_path, img_name))

    img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
    boundary = cv2.cvtColor(numpy.asarray(boundary), cv2.COLOR_RGB2BGR)
    mask = cv2.cvtColor(numpy.asarray(mask), cv2.COLOR_RGB2BGR)

    h, w = img.shape[0], img.shape[1]
    origin_coord = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]])

    theta = 30  # shear角度
    tan = math.tan(math.radians(theta))

    # x方向错切
    m = np.eye(3)
    m[0, 1] = tan
    shear_coord = (m @ origin_coord.T).T.astype(np.int64)

    shear_img = cv2.warpAffine(src=img, M=m[:2],
                               dsize=(np.max(shear_coord[:, 0]), np.max(shear_coord[:, 1])),
                               borderValue=(0, 0, 0))
    shear_boundary = cv2.warpAffine(src=boundary, M=m[:2],
                               dsize=(np.max(shear_coord[:, 0]), np.max(shear_coord[:, 1])),
                               borderValue=(0, 0, 0))
    shear_mask = cv2.warpAffine(src=mask, M=m[:2],
                               dsize=(np.max(shear_coord[:, 0]), np.max(shear_coord[:, 1])),
                               borderValue=(0, 0, 0))

    c_img = Image.fromarray(cv2.cvtColor(shear_img, cv2.COLOR_BGR2RGB))
    c_boundary = Image.fromarray(cv2.cvtColor(shear_boundary, cv2.COLOR_BGR2RGB))
    c_mask = Image.fromarray(cv2.cvtColor(shear_mask, cv2.COLOR_BGR2RGB))
    return c_img, c_boundary, c_mask


def hsv(root_path, img_name, boundary_path, mask_path):  # HSV数据增强
    h_gain, s_gain, v_gain = 0.5, 0.5, 0.5
    img = Image.open(os.path.join(root_path, img_name))
    boundary = Image.open(os.path.join(boundary_path, img_name))
    mask = Image.open(os.path.join(mask_path, img_name))

    img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)

    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    aug_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    aug_img = Image.fromarray(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))
    return aug_img, boundary, mask


def pingyi(root_path, img_name, boundary_path, mask_path):  # 平移扩充图像，根图像移动的像素距离可自行调整，具体方法如下注释所示
    img = Image.open(os.path.join(root_path, img_name))
    boundary = Image.open(os.path.join(boundary_path, img_name))
    mask = Image.open(os.path.join(mask_path, img_name))

    img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
    boundary = cv2.cvtColor(numpy.asarray(boundary), cv2.COLOR_RGB2BGR)
    mask = cv2.cvtColor(numpy.asarray(mask), cv2.COLOR_RGB2BGR)

    cols, rows = img.shape[0], img.shape[1]

    x = np.random.randint(30, 200)
    y = np.random.randint(30, 200)

    M = np.float32([[1, 0, x], [0, 1, y]])  # 50为x即水平移动的距离，30为y 即垂直移动的距离

    imgDst = cv2.warpAffine(img, M, (cols, rows), borderValue=(0, 0, 0))
    boundaryDst = cv2.warpAffine(boundary, M, (cols, rows), borderValue=(0, 0, 0))
    maskDst = cv2.warpAffine(mask, M, (cols, rows), borderValue=(0, 0, 0))

    pingyi_img = Image.fromarray(cv2.cvtColor(imgDst, cv2.COLOR_BGR2RGB))
    pingyi_boundary = Image.fromarray(cv2.cvtColor(boundaryDst, cv2.COLOR_BGR2RGB))
    pingyi_mask = Image.fromarray(cv2.cvtColor(maskDst, cv2.COLOR_BGR2RGB))
    return pingyi_img, pingyi_boundary, pingyi_mask


def createImage(
        imageDir, imageSave, boundaryDir, boundarySave, maskDir, maskSave
):  # 主函数，8种数据扩充方式，每种扩充一张
    i = 0
    for name in os.listdir(imageDir):
        i = i + 1
        print(name.rstrip('.tif'))

        saveName = name.rstrip('.tif') + '_pingyi.png'
        saveImage, saveBoundary, saveMask = pingyi(imageDir, name, boundaryDir, maskDir)
        saveImage.save(os.path.join(imageSave, saveName))
        saveBoundary.save(os.path.join(boundarySave, saveName))
        saveMask.save(os.path.join(maskSave, saveName))

        # saveName1 = "flip" + str(i) + ".jpg"
        # saveImage1 = flip(imageDir, name)
        # saveImage1.save(os.path.join(saveDir, saveName1))
        # saveName2 = "brightnessE" + str(i) + ".jpg"
        # saveImage2 = brightnessEnhancement(imageDir, name)
        # saveImage2.save(os.path.join(saveDir, saveName2))
        # saveName3 = "rotate" + str(i) + ".jpg"
        # saveImage = rotation(imageDir, name)
        # saveImage.save(os.path.join(saveDir, saveName3))
        # saveName4 = "fangshe" + str(i) + ".jpg"
        # saveImage = fangshe_bianhuan(imageDir, name)
        # saveImage.save(os.path.join(saveDir, saveName4))
        # saveName5 = "cuoqie" + str(i) + ".jpg"
        # saveImage = cuoqie(imageDir, name)
        # saveImage.save(os.path.join(saveDir, saveName5))
        # saveName6 = "hsv" + str(i) + ".jpg"
        # saveImage = hsv(imageDir, name)
        # saveImage.save(os.path.join(saveDir, saveName6))
        # saveName6 = "pingyi" + str(i) + ".jpg"  # 不需要平移变换的，可以注释掉 这三行代码 135 136 137行
        # saveImage = pingyi(imageDir, name)  # 不需要平移变换的，可以注释掉 这三行代码
        # saveImage.save(os.path.join(saveDir, saveName6))  # 不需要平移变换的，可以注释掉 这三行代码


if __name__ == '__main__':
    imageDir = r"E:\Xian-Jia He\GIS\coding\DDM-Net\data\Chongqing\train\img"  # 要改变的图片的路径文件夹  在当前文件夹下，建立文件夹即可
    boundaryDir = r'E:\Xian-Jia He\GIS\coding\DDM-Net\data\Chongqing\train\boundary'
    maskDir = r'E:\Xian-Jia He\GIS\coding\DDM-Net\data\Chongqing\train\mask'

    imageSave = r"E:\Xian-Jia He\GIS\coding\DDM-Net\data\Chongqing\train\augmentation\img/"  # 数据增强生成图片的路径文件夹
    boundarySave = r'E:\Xian-Jia He\GIS\coding\DDM-Net\data\Chongqing\train\augmentation\boundary'
    maskSave = r'E:\Xian-Jia He\GIS\coding\DDM-Net\data\Chongqing\train\augmentation\mask'

    createImage(imageDir, imageSave, boundaryDir, boundarySave, maskDir, maskSave)
