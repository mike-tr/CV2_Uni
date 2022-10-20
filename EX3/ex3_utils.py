import sys
from typing import List, Tuple

import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt
import numpy.linalg as linalg


def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 1111


def convertToGray(im: np.ndarray) -> np.ndarray:
    l = len(im.shape)
    imgG = im
    if l > 2:
        imgG = 0.3 * im[:, :, 0] + 0.59 * im[:, :, 1] + 0.11 * im[:, :,  2]
    return imgG


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size:
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    img1g = convertToGray(im1)
    img2g = convertToGray(im2)

    warp = int(win_size / 2)

    kerdx = np.array([[1, 0, -1]])
    kerdy = kerdx.T
    ix = cv2.filter2D(img2g, -1, kerdx, borderType=cv2.BORDER_REPLICATE)
    iy = cv2.filter2D(img2g, -1, kerdy, borderType=cv2.BORDER_REPLICATE)

    it = img2g - img1g

    pos = []
    duv = []

    warp_d = int(win_size / 2)
    warp_up = int((win_size + 1) / 2)
    height, width = img2g.shape

    for x in range(warp_d, width - warp_up, step_size):
        for y in range(warp_d, height - warp_up, step_size):
            ixr = ix[y - warp_d: y + warp_up,
                     x - warp_d: x + warp_up].flatten()
            iyr = iy[y - warp_d: y + warp_up,
                     x - warp_d: x + warp_up].flatten()
            itr = it[y - warp_d: y + warp_up,
                     x - warp_d: x + warp_up].flatten()

            xx = (ixr * ixr).sum()
            yy = (iyr * iyr).sum()
            xy = (iyr * ixr).sum()
            xt = (ixr * itr).sum()
            yt = (iyr * itr).sum()

            # print(xx, yy, xy, xt, yt)

            AtA = np.array([[xx, xy],
                            [xy, yy]])
            Atb = np.array([[-xt],
                            [-yt]])

            if linalg.cond(AtA) < 1/sys.float_info.epsilon:
                AtAi = linalg.inv(AtA)

                e1, e2 = linalg.eigvals(AtA)
                if e1 < 1 or e2 < 1:
                    continue

                if e1 <= e2:
                    if e2 / e1 >= 100:
                        continue
                else:
                    if e1 / e2 >= 100:
                        continue

                u, v = -AtAi @ Atb

                pos.append([x, y])
                duv.append([u[0], v[0]])
            else:
                continue
    # print("mean")
    # print("mean-u :", np.array(duv)[:, :1].mean())
    # print("mean-v :", np.array(duv)[:, 1:].mean())
    return np.array(pos), np.array(duv)


def getGausianKernel(kernel_size: np.ndarray) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param inImage: Input image
    :param kernelSize: Kernel size
    :return: The Blurred image
    """
    gausianKer: np.ndarray = cv2.getGaussianKernel(kernel_size, 0)
    gausianKer = gausianKer.reshape((-1, 1))
    gausianKer = gausianKer @ gausianKer.T

    return gausianKer


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """

    height, width, d = unpackImg(img)

    r = int(2 ** levels)

    nheight = r * int(height / r)
    nwidth = r * int(width / r)

    img = img[:nheight, :nwidth]

    kernel = getGausianKernel(5)

    arr = []
    for i in range(levels):
        imgTemp = cv2.filter2D(
            img, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        laplacian = img - imgTemp
        arr.append(laplacian)

        img = imgTemp[::2, ::2]
    arr.append(img)
    arr.reverse()
    return arr


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Resotrs the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """

    kernel = getGausianKernel(5) * 4

    imgl = lap_pyr[0]
    for i in range(1, len(lap_pyr)):
        imgl = gaussExpand(imgl, kernel)
        imgl = imgl + lap_pyr[i]
    return imgl


def unpackImg(img: np.ndarray):
    try:
        height, width = img.shape
        return height, width, 0
    except:
        height, width, d = img.shape
        return height, width, d


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """

    height, width, d = unpackImg(img)

    r = int(2 ** levels)

    nheight = r * int(height / r)
    nwidth = r * int(width / r)

    img = img[:nheight, :nwidth]

    kernel = getGausianKernel(5)

    arr = []
    arr.append(img)
    for i in range(levels):
        imgTemp = cv2.filter2D(
            img, -1, kernel, borderType=cv2.BORDER_REFLECT)
        img = imgTemp[::2, ::2]
        arr.append(img)

    return arr


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    """
    Expands a Gaussian pyramid level one step up
    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """

    h, w, l = unpackImg(img)
    if l == 0:
        arrp = np.zeros((h * 2, w * 2))
        arrp[::2, :: 2] = img
    else:
        arrp = np.zeros((h * 2, w * 2, l))
        arrp[::2, ::2, ::] = img
    return cv2.filter2D(arrp, -1, gs_k, borderType=cv2.BORDER_REFLECT101)


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """

    la = laplaceianReduce(img_1, levels)
    lb = laplaceianReduce(img_2, levels)

    gm = gaussianPyr(mask, levels)
    gm.reverse()

    kernel = getGausianKernel(5) * 4

    print(gm[0].shape)
    print(la[0].shape, lb[0].shape)

    blended = la[0] * gm[0] + lb[0] * (1 - gm[0])

    for i in range(1, levels + 1):
        blended = gaussExpand(blended, kernel)
        blended += la[i] * gm[i] + lb[i] * (1 - gm[i])

    naiveBlend = img_1 * mask + img_2 * (1 - mask)

    height, width, d = unpackImg(blended)
    naiveBlend = naiveBlend[: height, : width]
    return naiveBlend, blended
