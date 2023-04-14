import scipy
from scipy.signal import convolve2d
import numpy as np
from imageio import imread
from imageio import imwrite
from skimage.color import rgb2gray
from scipy.signal import convolve2d

MAX_PIXEL = 255
GRAYSCALE = 1


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img


def read_image(filename, representation):
  """
  Reads an image and converts it into a given representation
  :param filename: filename of image on disk
  :param representation: 1 for greyscale and 2 for RGB
  :return: Returns the image as an np.float64 matrix normalized to [0,1]
  """
  im = imread(filename)  # rgb return type
  if (representation == GRAYSCALE and im.ndim == 3):
    gray_scale_img = rgb2gray(im)
    return np.float64(gray_scale_img)
  else:

    return np.float64(im / MAX_PIXEL)


def reduce(im, blur_filter):
  """
  Reduces an image by a factor of 2 using the blur filter
  :param im: Original image
  :param blur_filter: Blur filter
  :return: the downsampled image
  """

  # blur the image to help us get rid of the high freq
  # sub sample - select only every 2nd pixel in very 2nd row
  blured_image = scipy.ndimage.convolve(im, blur_filter)
  col_blured_img = scipy.ndimage.convolve \
    (blured_image, np.transpose(blur_filter))
  return col_blured_img[::2, ::2]


def build_filter(filter_size):
    filter = [[1, 1]]
    conv = [[1, 1]]
    for i in range(filter_size - 2):
      filter = convolve2d(filter, conv)
    filter = filter / sum(filter[0])
    return filter

def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Builds a gaussian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    filter = build_filter(filter_size)
    pyr = [im]
    for i in range(max_levels - 1):
      temp_img = reduce(im, filter)
      if len(temp_img[0]) < 16 or len(temp_img) < 16:
        break
      else:
        pyr.append(temp_img)
        im = temp_img
    return pyr, filter
