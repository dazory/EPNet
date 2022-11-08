import numpy as np
from numpy import random
from PIL import Image
from torchvision import transforms
import torch



from PIL import Image, ImageOps, ImageEnhance

def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / 10)


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval.

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / 10.


def sample_level(n):
  return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _, __):
  return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _, __):
  return ImageOps.equalize(pil_img)


def posterize(pil_img, level, _):
  level = int_parameter(sample_level(level), 4)
  return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level, _):
  degrees = int_parameter(sample_level(level), 30)
  if np.random.uniform() > 0.5:
    degrees = -degrees
  return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level, _):
  level = int_parameter(sample_level(level), 256)
  return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level, img_size):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform(img_size,
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level, img_size):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform(img_size,
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level, img_size):
  level = int_parameter(sample_level(level), img_size[0] / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform(img_size,
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level, img_size):
  level = int_parameter(sample_level(level), img_size[1] / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform(img_size,
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level, _):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level, _):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level, _):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level, _):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)




"""
Augmix with pillow
"""

class AugMix():
    def __init__(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.to_rgb = True

        self.mixture_width = 3
        self.mixture_depth = -1

        self.aug_prob_coeff = 1.
        self.aug_severity = 1



        augmentations_without_obj_translation = [
            autocontrast, equalize, posterize, solarize,
            color, contrast, brightness, sharpness
        ]
        self.aug_list = augmentations_without_obj_translation


    def __call__(self, results):

        aug_img = self.aug(results)
        return aug_img




    # def __repr__(self):
    #     repr_str = self.__class__.__name__
    #     repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
    #     return repr_str


    def do_normalize(self, img_array):
        img_array = np.array(img_array, dtype=np.uint8)
        img_array = img_array / 255.0
        img_array -= self.mean
        img_array /= self.std
        img_tensor = transforms.ToTensor()(img_array).float()

        return img_tensor


    def aug(self, img):
        ws = np.float32(
            np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))
        m = np.float32(np.random.beta(self.aug_prob_coeff, self.aug_prob_coeff))
        IMAGE_HEIGHT, IMAGE_WIDTH, _ = img.shape
        img_size = (IMAGE_WIDTH, IMAGE_HEIGHT)

        img_tensor = np.transpose(img.copy(),(2,0,1))
        img_tensor = torch.Tensor(img_tensor)
        mix = torch.zeros_like(img_tensor)
        for i in range(self.mixture_width):
            image_aug = Image.fromarray(img.copy(), "RGB")
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op = np.random.choice(self.aug_list)
                image_aug = op(image_aug, self.aug_severity, img_size)

            mix += ws[i] * self.do_normalize(image_aug)
        mixed = (1 - m) * self.do_normalize(img) + m * mix
        mixed = mixed.numpy()
        mixed = np.transpose(mixed.copy(), (1, 2, 0))



        # mix = np.zeros_like(img.copy(), dtype=np.float32)
        # for i in range(self.mixture_width):
        #     image_aug = Image.fromarray(img.copy(), "RGB")
        #     depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
        #     for _ in range(depth):
        #         op = np.random.choice(self.aug_list)
        #         image_aug = op(image_aug, self.aug_severity, img_size)
        #
        #     image_aug = np.asarray(image_aug, dtype=np.float32)
        #     mix += ws[i] * image_aug
        # mixed = (1 - m) * img + m * mix
        # mixed = np.trunc(mixed)



        return mixed











