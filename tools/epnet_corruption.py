import time
import numpy as np

import skimage as sk
from skimage.filters import gaussian
from io import BytesIO
from wand.image import Image as WandImage
from wand.api import library as wandlibrary

from PIL import Image as PILImage
import cv2


class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


def motion_blur(x, severity=1):
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]

    output = BytesIO()
    x.save(output, format='PNG')
    x = MotionImage(blob=output.getvalue())

    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

    x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
                     cv2.IMREAD_UNCHANGED)

    if x.shape != (224, 224):
        return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
    else:  # greyscale to RGB
        return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def defocus_blur(x, severity=1):
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

    # x = np.array(x) / 255.
    x = x / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

    return np.clip(channels, 0, 1) * 255


def frost(x, severity=1):
    c = [(1, 0.4),
         (0.8, 0.6),
         (0.7, 0.7),
         (0.65, 0.7),
         (0.6, 0.75)][severity - 1]
    idx = np.random.randint(5)
    fileroot = "/ws/epnet_carla_ws/src/EPNet/tools/frost_img"
    filename = [fileroot+'/frost1.png', fileroot+'/frost2.png', fileroot+'/frost3.png', fileroot+'/frost4.jpg', fileroot+'/frost5.jpg', fileroot+'/frost6.jpg'][idx]
    frost = cv2.imread(filename)

    # print(x.shape[0])
    frost = cv2.resize(frost, (1500, 600))  # frost img size should be bigger than custom_img size
    # randomly crop and convert to rgb
    x_start, y_start = np.random.randint(0, frost.shape[0] - x.shape[0]), np.random.randint(0,
                                                                                            frost.shape[1] - x.shape[1])
    frost = frost[x_start:x_start + x.shape[0], y_start:y_start + x.shape[1]][..., [2, 1, 0]]

    return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)

def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1):
    c = [60, 25, 12, 5, 3][severity - 1]

    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255


def impulse_noise(x, severity=1):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255


def contrast(x, severity=1):
    c = [0.4, .3, .2, .1, .05][severity - 1]

    # x = np.array(x) / 255.
    x = x / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def brightness(x, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def jpeg_compression(x, severity=1):
    c = [25, 18, 15, 10, 7][severity - 1]

    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = PILImage.open(output)

    return x


def pixelate(x, severity=1):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]

    x = x.resize((int(224 * c), int(224 * c)), PILImage.BOX)
    x = x.resize((224, 224), PILImage.BOX)
    x = np.array(x)
    return x


def gen_corruption(x, method='frost', severity=1):
    if method == 'frost':
        result_np = frost(x, severity)  # 0.031
    elif method == 'contrast':
        result_np = contrast(x, severity)  # 0.027
    elif method == 'gaussian_noise':
        result_np = gaussian_noise(x, severity)  # 0.056
    elif method == 'shot_noise':
        result_np = shot_noise(x, severity)  # 0.16
    elif method == 'impulse_noise':
        result_np = impulse_noise(x, severity)  # 0.062
    elif method == 'motion_blur':
        result_np = motion_blur(x, severity)  # 0.468
    elif method == 'defocus_blur':
        result_np = defocus_blur(x, severity)  # 0.038
    elif method == 'brightness':
        result_np = brightness(x, severity)  # 0.165
    elif method == 'jpeg_compression':
        result_np = jpeg_compression(x, severity)  # 0.017
    elif method == 'pixelate':
        result_np = pixelate(x, severity)  # 0.0089

    return result_np

def gen_corruption_v2(x_np, method='frost', severity=1):
    x = PILImage.fromarray(x_np)
    # print(type(x))
    if method == 'frost':
        result_np = frost(x, severity)  # 0.031 done! & total mean time == 0.121
    elif method == 'contrast':
        result_np = contrast(x, severity)  # 0.027 done!  & total mean time == 0.125
    elif method == 'gaussian_noise':
        result_np = gaussian_noise(x, severity)  # 0.056 done& total mean time == 0.166 sec # deprecated
    elif method == 'shot_noise':
        result_np = shot_noise(x, severity)  # 0.16
    elif method == 'impulse_noise':
        result_np = impulse_noise(x, severity)  # 0.062
    elif method == 'motion_blur':
        result_np = motion_blur(x, severity)  # 0.468
    elif method == 'defocus_blur': # done & 0.157 sec
        result_np = defocus_blur(x, severity)  # 0.038
    elif method == 'brightness':
        result_np = brightness(x, severity)  # 0.165
    elif method == 'jpeg_compression':
        result_np = jpeg_compression(x, severity)  # 0.017
    elif method == 'pixelate':
        result_np = pixelate(x, severity)  # 0.0089

    return result_np
def main():
    img = PILImage.open('/home/ck/corruption/ImageNet-C/create_c/000265.png')
    # np -> PILImage object
    # img = PILImage.fromarray(np.uint8(numpy))
    start_time = time.time()
    np_img = gen_corruption(img, 'pixelate', 2)
    print(time.time() - start_time)

    PILImage.fromarray(np.uint8(np_img)).save('/home/ck/corruption/ImageNet-C/create_c/test.png')


if __name__ == "__main__":
    main()