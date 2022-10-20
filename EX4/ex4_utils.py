from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import ex3_utils as ex3
import cv2
from time import time


def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 1111


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


def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: Tuple[int, int], k_size: int) -> np.ndarray:
    """

    img_l: Left image
    img_r: Right image
    range: Minimun and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    height, width = img_r.shape
    img_l: np.ndarray = np.pad(img_l,  ((0, 0),
                                        (0, disp_range[1])), mode='edge')

    disp = np.zeros(img_r.shape)

    t1 = time()
    for x in range(k_size, width - k_size):
        for y in range(k_size, height - k_size):
            windowl: np.ndarray = img_r[y - k_size: y +
                                        k_size + 1, x - k_size: x + k_size + 1]
            windowl = windowl.flatten()
            best = 9999999
            disparity = -1
            for offset in range(disp_range[0], disp_range[1]):
                windowr: np.ndarray = img_l[y - k_size: y + k_size + 1, x +
                                            offset - k_size: x + offset + k_size + 1]

                windowr = windowr.flatten()
                # print(windowr.shape, "p : ", x, y, "s2",
                #       windowl.shape, "off :", offset, img_r.shape)
                curr = np.square(windowl - windowr).sum()

                #curr = -windowr.sum()
                if(curr < best):
                    best = curr
                    disparity = offset
            disp[y, x] = disparity
            #print(best, best_off)

    #disp /= disp.max()
    t2 = time()
    #print(t2 - t1)
    return disp


def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: Tuple[int, int], k_size: int) -> np.ndarray:
    """

    img_l: Left image
    img_r: Right image
    range: Minimun and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    height, width = img_r.shape
    img_l: np.ndarray = np.pad(img_l,  ((0, 0),
                                        (0, disp_range[1])), mode='edge')

    disp = np.zeros(img_r.shape)

    t1 = time()
    for x in range(k_size, width - k_size):
        for y in range(k_size, height - k_size):
            windowl: np.ndarray = img_r[y - k_size: y +
                                        k_size + 1, x - k_size: x + k_size + 1]
            windowl = windowl.flatten()
            sl = np.square(windowl).sum()
            best = -1
            disparity = -1
            for offset in range(disp_range[0], disp_range[1]):
                windowr: np.ndarray = img_l[y - k_size: y + k_size + 1, x +
                                            offset - k_size: x + offset + k_size + 1]

                windowr = windowr.flatten()
                top = (windowr @ windowl)
                sr = np.square(windowr).sum()

                # print(top)
                #print(sr.shape, sl.shape)
                buttom = np.sqrt(sr * sl)
                # print(windowr.shape, "p : ", x, y, "s2",
                #       windowl.shape, "off :", offset, img_r.shape)
                curr = top / buttom

                #curr = -windowr.sum()
                if(curr > best):
                    best = curr
                    disparity = offset
            disp[y, x] = disparity
            #print(best, best_off)

    #disp /= disp.max()
    t2 = time()
    #print(t2 - t1)
    return disp


def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> Tuple[np.ndarray, float]:
    """
        Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
        returns the homography and the error between the transformed points to their
        destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))

        src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
        dst_pnt: 4+ keypoints locations (x,y) on the destenation image. Shape:[4+,2]

        return: (Homography matrix shape:[3,3],
                Homography error)
    """

    height, width = src_pnt.shape
    A = np.zeros((height * 2, 9))

    # print(src_pnt[2, 1])
    for row in range(height):
        xit = dst_pnt[row, 0]
        yit = dst_pnt[row, 1]
        xi = src_pnt[row, 0]
        yi = src_pnt[row, 1]

        xar = row * 2
        yar = xar + 1

        A[xar] = [xi, yi, 1, 0, 0, 0, -xit * xi, -xit * yi, -xit]
        A[yar] = [0, 0, 0, xi, yi, 1, -yit*xi, -yit*yi, -yit]

    s, v, d = np.linalg.svd(A)
    # v1 = np.zeros(9)
    # v1[:-1] = v
    # vd = np.diag(v1)
    # vd = vd[:-1, :]
    # print(A)
    # print("reconstructed")
    # print(s @ vd @ d)

    h = d[8]
    h = h.reshape((3, 3))
    e = h[2, 2]
    h /= e
    # print(h, e)

    # src_pnt = np.array([src_pnt[0]])
    # print(src_pnt)
    pred = translateByHomography(src_pnt, h)
    # print("\npred not n")
    # print(pred)

    err = pred - dst_pnt
    err = np.abs(err).mean()
    # print(err)

    return h, err


def translateByHomography(src_pnt, h):
    X = np.hstack([src_pnt, np.ones([src_pnt.shape[0], 1])])
    pred = h @ X.T

    dv = pred[-1]
    dv = dv.reshape(-1, 1)
    pred = pred.T / dv
    return pred[:, :-1]


def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
       Displays both images, and lets the user mark 4 or more points on each image. Then calculates the homography and transforms the source image on to the destination image. Then transforms the source image onto the destination image and displays the result.

       src_img: The image that will be 'pasted' onto the destination image.
       dst_img: The image that the source image will be 'pasted' on.

       output:
        None.
    """

    dst_p = []
    fig1 = plt.figure()

    def onclick_1(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        dst_p.append([x, y])

        if len(dst_p) == 4:
            plt.close()
        plt.show()

    # display image 1
    cid = fig1.canvas.mpl_connect('button_press_event', onclick_1)
    plt.imshow(dst_img)
    plt.show()
    dst_p = np.array(dst_p)

    ##### Your Code Here ######

    #print(src_img.shape, src_img[:, :].shape)
    r = src_img[:, :, 0]
    g = src_img[:, :, 1]
    b = src_img[:, :, 2]
    height, width = r.shape

    src_vec = np.array([[0, 0],
                        [width, 0],
                        [width, height],
                        [0, height]])
    print(height, width)

    print(src_vec)
    print(dst_p)

    h, e = computeHomography(src_vec, dst_p)

    cords = np.array(np.meshgrid(range(width), range(height))).T.reshape(-1, 2)
    #print(cords.shape, cords)
    #tpos = translateByHomography(cords, h)

    v = np.array([0, 0, 1])
    print(h @ v)
    #print(v, translateByHomography(v, h))
    for x in range(width):
        for y in range(height):
            v[0] = x
            v[1] = y
            n = h @ v
            n /= n[2]
            n = n[:-1]
            cn = n.astype(int)
            #print("??? ",  src_img[x, y, :])
            #print("???2 ", x, y, cn[0], cn[1])
            dst_img[cn[1], cn[0], :] = src_img[y, x, :]
            #print(cn - n)

            # dst_img[x, y] =
            # print(rt)

    plt.imshow(dst_img)
    plt.show()


def warpImagAdvanced(src_img: np.ndarray, dst_img: np.ndarray, src_mask: np.ndarray) -> None:
    """
       Displays both images, and lets the user mark 4 or more points on each image. Then calculates the homography and transforms the source image on to the destination image. Then transforms the source image onto the destination image and displays the result.

       src_img: The image that will be 'pasted' onto the destination image.
       dst_img: The image that the source image will be 'pasted' on.
       src_mask: in black everything u do not want from src_img, and in white everything u want
       then we scale the mask too and then use blending.

       output:
        None.
    """

    dst_p = []
    fig1 = plt.figure()

    def onclick_1(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        dst_p.append([x, y])

        if len(dst_p) == 4:
            plt.close()
        plt.show()

    print("click 4 points on screen");
    # display image 1
    cid = fig1.canvas.mpl_connect('button_press_event', onclick_1)
    plt.imshow(dst_img)
    plt.show()
    dst_p = np.array(dst_p)

    ##### Your Code Here ######

    #print(src_img.shape, src_img[:, :].shape)
    r = src_img[:, :, 0]
    g = src_img[:, :, 1]
    b = src_img[:, :, 2]
    height, width = r.shape

    src_vec = np.array([[0, 0],
                        [width, 0],
                        [width, height],
                        [0, height]])


    h, e = computeHomography(src_vec, dst_p)
    v = np.array([0, 0, 1])

    nmask = np.zeros(dst_img.shape)
    mapped = dst_img.copy()
    # print(h @ v)
    #print(v, translateByHomography(v, h))
    for x in range(width):
        for y in range(height):
            v[0] = x
            v[1] = y
            n = h @ v
            n /= n[2]
            n = n[:-1]
            cn = n.astype(int)
            #print("??? ",  src_img[x, y, :])
            #print("???2 ", x, y, cn[0], cn[1])
            mapped[cn[1], cn[0], :] = src_img[y, x, :]
            nmask[cn[1], cn[0], :] = src_mask[y, x, :]
            #print(cn - n)

            # dst_img[x, y] =
            # print(rt)

    # plt.imshow(dst_img)
    # plt.imshow(nmask)
    nb, blended = ex3.pyrBlend(mapped, dst_img, nmask, 5)
    plt.imshow(blended)
    plt.show()
