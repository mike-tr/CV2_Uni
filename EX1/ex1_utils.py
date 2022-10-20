"""
        '########:'##::::'##::::'##:::
         # .....::. ##::'##:::'####:::
         # ::::::::. ##'##::::.. ##:::
         # :::::. ###::::::: ##:::
         # ...:::::: ## ##:::::: ##:::
         # :::::::: ##:. ##::::: ##:::
         # : ##:::. ##::'######:
        ........::..:::::..:::......::
"""
import imp
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import time

from numpy.core.fromnumeric import shape
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def _checkViableImage(image: np.ndarray) -> bool:
    l = len(image.shape)
    if l == 2:
        return True
    if l == 3:
        if image.shape[2] != 3:
            print("Image third dimention must be of size 3")
            return False
        return True
    print("given input cannot be an image the shape is not of the type (width, height, 3) or (width, height)")
    return False


def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 111111


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE (1) or RGB (2)
    :return: The image object

    GRAY_SCALE : we will take ( (0.3 * R) + (0.59 * G) + (0.11 * B) ).
    Note : Image loaded as BGR and not RGB.

    return image as float between [0, 1]
    """

    try:
        image: np.ndarray = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255
        if representation == 2:
            return image
        elif representation == 1:
            return 0.3 * image[:, :, 0] + 0.59 * image[:, :, 1] + 0.11 * image[:, :,  2]
    except:
        print("something went wrong perhabs the image does not exist!")


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE (1) or RGB (2)
    :return: None
    """
    im = imReadAndConvert(filename, representation)
    if im is not None:
        imgDisplay(im)


def imgDisplay(img: np.ndarray):
    if _checkViableImage(img) is False:
        return
    l = len(img.shape)
    if l == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space

    after some test i now know how to multiply image by a matrix.
    """
    if _checkViableImage(imgRGB) is False:
        return
    # we need to multiply the matricies.
    mat = np.array([[0.299, 0.587, 0.114],
                    [0.596, -0.275, -0.321],
                    [0.212, -0.523, 0.311]])
    return imgRGB @ mat.T


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    if _checkViableImage(imgYIQ) is False:
        return
    # straight forward the ivnerse of YIQ
    mat = np.array([[0.299, 0.587, 0.114],
                    [0.596, -0.275, -0.321],
                    [0.212, -0.523, 0.311]])
    mat = np.linalg.inv(mat)
    imRGB = imgYIQ @ mat.T
    return imRGB


def imHistogram(img: np.ndarray) -> np.ndarray:
    """
        This is internal function create a histogram of a given image.
        img  : image in grayscale a.k.a 2D, with ranges [0, 255]
    """
    histo = np.zeros((256), dtype=int)
    for column in img:
        for pixel in column:
            histo[pixel] += 1
    return histo


def hsitogramEqualize(imgOrig: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    if _checkViableImage(imgOrig) is False:
        return

    # Convert any image to grayscale 256, save the original YIQ if its RGB
    l = len(imgOrig.shape)
    if l == 3:
        imYIQ = transformRGB2YIQ(imgOrig)
        im256 = imYIQ[:, :, 0]
        im256: np.ndarray = np.round(im256 * 255).astype(int)
    else:
        im256: np.ndarray = np.round(imgOrig * 255).astype(int)
    original_histo = imHistogram(im256)

    # calculate how many pixels in the image, a.k.a height * width
    numpixels = im256.shape[0] * im256.shape[1]

    # create lut and cumsum
    cumsum = np.zeros((256), dtype=int)
    lut = np.zeros((256), dtype=int)
    cumsum[0] = original_histo[0]
    for i in range(1, 256):
        cumsum[i] = cumsum[i - 1] + original_histo[i]
    lut = np.ceil(cumsum * 255 / numpixels).astype(int)

    imgEq = np.zeros(im256.shape, dtype=int)

    # honestly it might be Slower to do it imgEq[img256 == intensity] , in this case we will loop over the image 256 times!
    # so there is No benefit at all in making it 1 loop instead of 2.
    for x in range(0, imgEq.shape[0]):
        for y in range(0, imgEq.shape[1]):
            intensity = im256[x][y]
            imgEq[x][y] = lut[intensity]

    Eq_histogram = imHistogram(imgEq)
    # print(imgEq.min(), imgEq.max())
    imgEq = imgEq / 255
    # print(imgEq.min(), imgEq.max())
    if l == 3:
        # convert back to RGB
        imYIQ[:, :, 0] = imgEq
        img = transformYIQ2RGB(imYIQ)
        return img, original_histo, Eq_histogram
    return imgEq, original_histo, Eq_histogram


def quantizeImage(imgOrig: np.ndarray, nQuant: int, nIter: int) -> Tuple[List[np.ndarray], List[float]]:
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    if _checkViableImage(imgOrig) is False:
        return

    # Same if its RGB convert to YIQ and use the Y channel, else use the grayscale.
    l = len(imgOrig.shape)
    imYIQ = None
    if l == 3:
        imYIQ = transformRGB2YIQ(imgOrig)
        im256 = imYIQ[:, :, 0]
        im256: np.ndarray = np.ceil(im256 * 255).astype(int)
    else:
        im256: np.ndarray = np.ceil(imgOrig * 255).astype(int)

    # Initialize the Q's and Z's
    numpixels = im256.shape[0] * im256.shape[1]
    original_histo = imHistogram(im256)
    z = np.zeros(nQuant + 1, dtype=int)
    q = np.zeros(nQuant, dtype=int)
    z[0] = 0
    z[nQuant] = 255
    j = 1

    # Create cumsum, because this is a more efficient Way.
    cumsum = np.zeros((256), dtype=int)
    cumsum[0] = original_histo[0]
    for i in range(1, 256):
        cumsum[i] = cumsum[i - 1] + original_histo[i]

    t = numpixels / nQuant
    for i in range(1, 256):
        if cumsum[i] > t * j:
            # print(j, i, cumsum[i]-cumsum[z[j-1]])
            z[j] = i
            j += 1
            if j == nQuant:
                break
    zrange = np.arange(256)

    # Create internal function for calculating Q, given Z
    def updateQ():
        for i in range(nQuant):

            zmin = z[i]
            zmax = z[i+1]

            if i + 1 == nQuant:
                zmax = 256
            if zmin == 0:
                cmin = 0
            else:
                cmin = cumsum[zmin - 1]

            pixels = cumsum[zmax - 1] - cmin
            px = original_histo[zmin:zmax]
            intensities = zrange[zmin:zmax]
            q[i] = (px @ intensities) / pixels
            # print(i, zmin, zmax, q[i])

    # calcualte Z values, given Q
    def updateZ():
        for i in range(1, nQuant - 1):
            # print(i, q[i], q[i-1])
            z[i] = (q[i] + q[i-1]) / 2

    # method for calculating Error, on the Histogran, rather then on the image itself
    # this is much much faster then using the second method with is only 1 line of code!
    # I TESTED IT, its about 10 times as fast, and its probably even faster on big pictures.
    def calcErr() -> float:
        delta = 0
        for i in range(nQuant):

            zmin = z[i]
            zmax = z[i+1]

            if zmin == 0:
                cmin = 0
            else:
                cmin = cumsum[zmin - 1]

            pixels = cumsum[zmax - 1] - cmin
            px = original_histo[zmin:zmax]
            intensities = zrange[zmin:zmax]

            qdelta = ((intensities - q[i]) ** 2)
            # print(qdelta)
            # print(i, zmin, zmax - 1, q[i], "\n", qdelta)
            qdelta = qdelta @ px
            # print(qdelta)
            delta += qdelta
        # print(math.sqrt(delta) / numpixels)
        return math.sqrt(delta) / numpixels

    # Create a quantalized image.
    def quantImg() -> np.ndarray:
        qImg = np.zeros(im256.shape)
        for i in range(nQuant):
            # print(z[i], z[i+1], q[i])
            select = (im256 >= z[i]) & (im256 < z[i+1])
            qImg[select] = q[i]
        return qImg

    # # Very bas way of doing what Err doing, but "the easy way", of course both fucntion return exactly the same value!
    # def calcErr2() -> float:
    #     #  this is the one liner we were told we can do, but its about 10 times slower ( even if we have the quntilized image ).
    #     #  then the Err1 method
    #     qImg = quantImg()
    #     # t0 = time.time()
    #     err = math.sqrt(((qImg - im256) ** 2).sum())
    #     # print(err / numpixels)
    #     # print("e2 ", time.time() - t0)
    #     return err

    # do nIterations and save Err and QImage in arr.
    z[nQuant] = 256
    arr_qim = []
    arr_errors = []
    for i in range(nIter):
        updateQ()
        updateZ()
        # t0 = time.time()
        if(imYIQ is not None):
            qim = imYIQ.copy()
            # print(quantImg().max())
            qim[:, :, 0] = (quantImg() / 255)
            qim = transformYIQ2RGB(qim)
        else:
            qim = quantImg()
        arr_qim.append(qim)
        arr_errors.append(calcErr())
    return arr_qim, arr_errors
