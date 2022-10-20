from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from itertools import chain
import time

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


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


def imgDisplay(img: np.ndarray):
    l = len(img.shape)
    if l == 2:
        return plt.imshow(img, cmap='gray')
    else:
        return plt.imshow(img)


def conv1D(inSignal: np.ndarray, kernel1: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param inSignal: 1-D array
    :param kernel1: 1-D array as a kernel
    :return: The convolved array
    """
    size_in = inSignal.shape[0]
    size_ker = kernel1.shape[0]
    output = np.zeros((size_in + size_ker - 1))
    for n in range(output.shape[0]):
        for m in range(size_ker):
            x = n - m
            p = 0
            if 0 <= x < size_in:
                p = inSignal[x]
            output[n] += p * kernel1[m]
    return output


def conv2D(inImage: np.ndarray, kernel2: np.ndarray) -> np.ndarray:
    size_in_y = inImage.shape[0]
    size_in_x = inImage.shape[1]
    size_ker_y = kernel2.shape[0]
    size_ker_x = kernel2.shape[1]

    output = np.zeros((inImage.shape))

    offset_y = int((kernel2.shape[0] - 1) / 2)
    offset_x = int((kernel2.shape[1] - 1) / 2)

    im: np.ndarray = np.pad(inImage,  ((size_ker_y - 1 - offset_y, offset_y),
                                       (size_ker_x - 1 - offset_x, offset_x)), mode='reflect')

    k = kernel2.ravel()

    for iy in range(size_in_y):
        for ix in range(size_in_x):
            area = im[iy:iy + size_ker_y, ix: ix + size_ker_x].flatten()
            output[iy, ix] = area.dot(k)
    return output


def convDerivative(inImage: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate gradient of an image
    :param inImage: Grayscale iamge
    :return: (directions, magnitude,x_der,y_der)
    """
    kerdx = np.array([[1, 0, -1]])
    kerdy = kerdx.T
    dx = conv2D(inImage, kerdx)
    dy = conv2D(inImage, kerdy)

    magnitude = dx * dx + dy * dy
    magnitude = np.sqrt(magnitude)

    dxc = dx.copy()
    dxc[dxc == 0] = 0.0000001

    directions = np.arctan(dy/dxc)
    return (directions, magnitude, dx, dy)


def _getGausianKernel(kernel_size: int) -> np.ndarray:
    ker = np.array([1, 1])
    gausianKer = np.array([1, 1])
    for i in range(kernel_size - 2):
        gausianKer = conv1D(gausianKer, ker)
    gausianKer = gausianKer.reshape(-1, 1)
    gausianKer = gausianKer @ gausianKer.T
    gausianKer /= gausianKer.sum()

    return gausianKer


def blurImage1(in_image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param inImage: Input image
    :param kernelSize: Kernel size
    :return: The Blurred image
    """
    gausianKer = _getGausianKernel(kernel_size)
    return conv2D(in_image, gausianKer)


def blurImage2(in_image: np.ndarray, kernel_size: np.ndarray) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param inImage: Input image
    :param kernelSize: Kernel size
    :return: The Blurred image
    """
    gausianKer: np.ndarray = cv2.getGaussianKernel(kernel_size, 0)
    gausianKer = gausianKer.reshape((-1, 1))
    gausianKer = gausianKer @ gausianKer.T

    return cv2.filter2D(in_image, -1, gausianKer)


def _sobleDir(img: np.ndarray):
    kernel_smooth = np.array([[1],
                              [2],
                              [1]])
    kernel_derivitive = np.array([[1, 0, -1]])
    kernel_dx: np.ndarray = kernel_smooth @ kernel_derivitive

    kernel_dx = kernel_dx
    kernel_dy = kernel_dx.T

    dx = conv2D(img, kernel_dx)
    dy = conv2D(img, kernel_dy)
    return dx, dy


def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects edges using the Sobel method
    :param img: Input image
    :param thresh: The minimum threshold for the edge response
    :return: opencv solution, my implementation

    we need to use smoothing in the x direction and find derivitive in y direction
    and do the same thing for x derivitive
    """
    dx, dy = _sobleDir(img)

    magnitude = np.sqrt(dx * dx + dy * dy)
    magnitude[magnitude > thresh] = 1
    magnitude[magnitude <= thresh] = 0

    sobelx: np.ndarray = cv2.Sobel(img, -1, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, -1, 0, 1, ksize=3)

    sobel_cv = np.sqrt(sobelx * sobelx + sobely * sobely)
    sobel_cv[sobel_cv > thresh] = 1
    sobel_cv[sobel_cv <= thresh] = 0
    return (sobel_cv, magnitude)


def _calculateLaplacian(laplacian_img: np.ndarray) -> np.ndarray:
    edges = np.zeros(laplacian_img.shape)
    height, width = laplacian_img.shape
    for x in range(1, width):
        for y in range(1, height):
            if laplacian_img[y, x - 1] > 0:
                if laplacian_img[y, x] == 0:
                    try:
                        if laplacian_img[y + 1, x] < 0:
                            edges[y, x] = 1
                    except:
                        edges[y, x] = 1
                if laplacian_img[y, x] < 0:
                    edges[y, x] = 1
            if laplacian_img[y-1, x] > 0:
                if laplacian_img[y, x] == 0:
                    try:
                        if laplacian_img[y+1, x] < 0:
                            edges[y, x] = 1
                    except:
                        edges[y, x] = 1
                if laplacian_img[y, x] < 0:
                    edges[y, x] = 1
            # other order
            if laplacian_img[y, x - 1] < 0:
                if laplacian_img[y, x] == 0:
                    try:
                        if laplacian_img[y + 1, x] > 0:
                            edges[y, x] = 1
                    except:
                        edges[y, x] = 1
                if laplacian_img[y, x] > 0:
                    edges[y, x] = 1
            if laplacian_img[y-1, x] < 0:
                if laplacian_img[y, x] == 0:
                    try:
                        if laplacian_img[y+1, x] > 0:
                            edges[y, x] = 1
                    except:
                        edges[y, x] = 1
                if laplacian_img[y, x] > 0:
                    edges[y, x] = 1
    return edges


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
        Detecting edges using the "ZeroCrossing" method
        :param img: Input image
        :return: Edge matrix
    """
    laplace_kernel = np.array([[0, -1, 0],
                               [-1, 4, -1],
                               [0, -1, 0]])
    limage = conv2D(img, laplace_kernel)
    # now we need to find where its goes from positive to negative

    return _calculateLaplacian(limage)
    # return limage
    # limage[limage > 0] = 0
    # limage[limage < 0] = 1
    # return limage


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> (np.ndarray):
    """
    Detecting edges using the "ZeroCrossingLOG" method
    :param img: Input image
    :return: :return: Edge matrix
    """
    bimage = blurImage2(img, 21)
    return edgeDetectionZeroCrossingSimple(bimage)


def deg2Rad(x):
    x = x * math.pi / 180
    return x


def simpleEdgeDetection(img: np.ndarray, thrs_1: float, kernel_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detecting edges usint "Canny Edge" method
    :param img: Input image
    :param thrs_1: T1
    :param thrs_2: T2
    :return: opencv solution, my implementation
    """
    simg = blurImage1(img, kernel_size)
    (directions, magnitude, dx, dy) = convDerivative(simg)

    edges = magnitude
    edges[edges > thrs_1] = 1
    edges[edges < 1] = 0

    return edges


def anyIsStrong(image, y, x):
    if image[y + 1, x] == 1 or image[y - 1, x] == 1 or image[y, x+1] == 1 or image[y, x-1] == 1 \
            or image[y - 1, x - 1] == 1 or image[y + 1, x + 1] == 1 or image[y - 1, x+1] == 1 or image[y + 1, x - 1] == 1:
        return True
    return False


def hysteresis(image):
    height, width = image.shape

    left_buttom = image.copy()
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            if 0 < left_buttom[y, x] < 1:
                if anyIsStrong(left_buttom, y, x):
                    left_buttom[y, x] = 1
                else:
                    left_buttom[y, x] = 0

    right_top = image.copy()
    for x in range(width - 2, 0, -1):
        for y in range(height - 2, 0, -1):
            if 0 < right_top[y, x] < 1:
                if anyIsStrong(right_top, y, x):
                    right_top[y, x] = 1
                else:
                    right_top[y, x] = 0

    left_top = image.copy()
    for x in range(1, width - 1):
        for y in range(height - 2, 0, -1):
            if 0 < left_top[y, x] < 1:
                if anyIsStrong(left_top, y, x):
                    left_top[y, x] = 1
                else:
                    left_top[y, x] = 0

    right_buttom = image.copy()
    for x in range(width - 2, 0, -1):
        for y in range(1, height - 1):
            if 0 < right_buttom[y, x] < 1:
                if anyIsStrong(right_buttom, y, x):
                    right_buttom[y, x] = 1
                else:
                    right_buttom[y, x] = 0

    img = right_top + left_buttom + right_buttom + left_top
    img[img > 1] = 1
    return img


def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detecting edges usint "Canny Edge" method
    :param img: Input image
    :param thrs_1: T1
    :param thrs_2: T2
    :return: opencv solution, my implementation
    """
    simg = blurImage2(img, 7)

    # (directions, magnitude, dx, dy) = convDerivative(simg)

    (dx, dy) = _sobleDir(simg)

    dxc = dx.copy()
    dxc[dxc == 0] = 0.0001
    directions: np.ndarray = np.arctan(dy, dxc)
    magnitude: np.ndarray = np.sqrt(dx * dx + dy * dy)

    height, width = img.shape

    edges = magnitude.copy()
    # edges = np.round(magnitude * 255).astype(int)

    directions = np.rad2deg(directions)
    directions[directions < 0] += 180

    # edges = np.zeros(magnitude.shape)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # print(y, x)
            angle = directions[y, x]
            mag = magnitude[y, x]

            if angle < 22.5 or angle > 157.5:
                # magn1 = edges[]
                n1_x, n1_y = x + 1, y
                n2_x, n2_y = x - 1, y

                # mag *= 1.05
            elif angle <= 67.5:
                n1_x, n1_y = x - 1, y - 1
                n2_x, n2_y = x + 1, y + 1
                # mag *= 2
            elif angle <= 112.5:
                n1_x, n1_y = x, y + 1
                n2_x, n2_y = x, y - 1
                # mag *= 1.05
            elif angle <= 157.5:
                # mag *= 2
                n1_x, n1_y = x - 1, y + 1
                n2_x, n2_y = x + 1, y - 1

            if mag < magnitude[n1_y, n1_x] or mag < magnitude[n2_y, n2_x]:
                magnitude[y, x] -= 0.005
                edges[y, x] = 0

    magnitude = edges.copy()

    edges[edges < thrs_2] = 0
    edges[edges >= thrs_1] = 1

    edges = hysteresis(edges)

    cimg: np.ndarray = (img * 255).astype(np.uint8)

    img_blur = blurImage2(cimg, 5)
    detected_edges = cv2.Canny(img_blur, thrs_2 * 255, thrs_1 * 255)

    return detected_edges, edges


def maskCircle(x0, y0, radius, height, width):
    x_min = x0 - radius
    y_min = y0 - radius
    x_max = x0 + radius + 1
    y_max = y0 + radius + 1

    if(x_min < 0):
        x_min = 0
    if(x_max > width):
        x_max = width
    if(y_min < 0):
        y_min = 0
    if(y_max > height):
        y_max = height

    x_ = np.arange(x_min, x_max, dtype=int)
    y_ = np.arange(y_min, y_max, dtype=int)

    mask_bigger = (x_[np.newaxis, :] - x0)**2 + \
        (y_[:, np.newaxis]-y0)**2 <= (radius + 1)**2

    mask_smaller = (x_[np.newaxis, :] - x0)**2 + \
        (y_[:, np.newaxis]-y0)**2 >= (radius - 1)**2

    return mask_bigger & mask_smaller, x_min, y_min, x_max, y_max


def fitCircleMask(x0, y0, width, height, radius, circle):
    x_min = x0 - radius - 1
    y_min = y0 - radius - 1
    x_max = x0 + radius + 2
    y_max = y0 + radius + 2

    x_start = 0
    x_end = radius * 2 + 3
    y_start = 0
    y_end = radius * 2 + 3

    if(x_min < 0):
        x_start -= x_min
        x_min = 0
    if(x_max > width):
        x_end += width - (x_max)
        x_max = width
    if(y_min < 0):
        y_start -= y_min
        y_min = 0
    if(y_max > height):
        y_end += height - (y_max)
        y_max = height

    return circle[y_start:y_end, x_start: x_end], x_min, y_min, x_max, y_max
    # return circle[]


def createCircleMask(radius):
    x = np.arange(0, radius * 2 + 3, dtype=int)
    y = np.arange(0, radius * 2 + 3, dtype=int)

    x0 = radius + 1
    y0 = radius + 1

    mask_bigger = (x[np.newaxis, :] - x0)**2 + \
        (y[:, np.newaxis]-y0)**2 <= (radius + 1)**2

    mask_smaller = (x[np.newaxis, :] - x0)**2 + \
        (y[:, np.newaxis]-y0)**2 >= (radius - 1)**2

    return mask_bigger & mask_smaller


def houghCircle(img: np.ndarray, min_radius: float, max_radius: float) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    :param I: Input image
    :param minRadius: Minimum circle radius
    :param maxRadius: Maximum circle radius
    :return: A list containing the detected circles,
    [(x,y,radius),(x,y,radius),...]

    i assume the radius is in pixels
    """
    edges_cv, edges_mine = edgeDetectionCanny(img, 0.5, 0.35)

    #tcircle = createCircleMask(7)
    # print(fitCircleMask(0, 49, 40, 50, 7, tcircle)[0])

    # print(createCircleMask(3))

    min_radius = int(min_radius)
    max_radius = int(max_radius + 0.5)

    circles = {}
    circle_masks = {}
    for i in range(min_radius, max_radius + 1):
        circles[i] = np.zeros(img.shape)
        circle_masks[i] = createCircleMask(i)

    height, width = img.shape
    arr_circles = []

    t0 = time.time()
    for y in range(height):
        for x in range(width):
            if edges_cv[y, x] == 255:
                for r in range(min_radius, max_radius + 1):
                    circle_mask, x_min, y_min, x_max, y_max = fitCircleMask(
                        x, y, width, height, r, circle_masks[r])
                    circles[r][y_min:y_max, x_min:x_max][circle_mask] += 1

    print("time to do new way : ", time.time() - t0)
    # t0 = time.time()
    # for y in range(height):
    #     for x in range(width):
    #         if edges_cv[y, x] == 255:
    #             for r in range(min_radius, max_radius + 1):
    #                 circle_mask, x_min, y_min, x_max, y_max = maskCircle(
    #                     x, y, r, height, width)
    #                 circles[r][y_min:y_max, x_min:x_max][circle_mask] += 1
    # print("time to do old way : ", time.time() - t0)

    for r in range(min_radius, max_radius + 1):
        for y, x in np.argwhere(circles[r] > 5*r):
            r2 = int(r/2) + 1

            area = circles[r][y-r2:y+r2, x-r2:x+r2]
            xt, yt = np.where(area == area.max())

            xm = xt[0] - r2 + x
            ym = yt[0] - r2 + y

            if(xm == x and ym == y):
                circles[r][y, x] += 1
                arr_circles.append((x, y, r))

    return arr_circles
