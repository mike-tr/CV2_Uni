from numpy.core.numeric import full
import ex2_utils as cv_mine
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

# folder = "images/"
folder = ""


def plot_images(images: list, labels: list):
    fig, axs = plt.subplots(1, len(images), figsize=(24, 6))

    for i in range(len(images)):
        plt.subplot(axs[i])
        cv_mine.imgDisplay(images[i])
        # plt.figure(figsize=(8, 8))
        plt.title(labels[i])
        plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()


def Show1DConv():
    print("----------- 1D conv ----------")
    signal = np.array([2, 1, 0, 0, 0, 1, 2])
    kernel = np.array([2, 0, 1])

    convolved = cv_mine.conv1D(signal, kernel)
    print("signal :", signal, "\nkernel :", kernel)
    print("my output :", convolved.astype(int))

    print("np output :", np.convolve(signal, kernel, 'full'))
    print()


def Show2DConv():
    kernel = np.random.random((7, 7)) / 49

    img = cv_mine.imReadAndConvert(
        folder + 'wtf_im.jpg', cv_mine.LOAD_GRAY_SCALE)

    dst = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)

    dst_mine = cv_mine.conv2D(img, kernel)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3))
    plt.subplot(ax1), cv_mine.imgDisplay(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(ax2), cv_mine.imgDisplay(dst_mine), plt.title('My conv')
    plt.xticks([]), plt.yticks([])
    plt.subplot(ax3), cv_mine.imgDisplay(dst), plt.title('CV2')
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()


def showDirEdges():
    img = cv_mine.imReadAndConvert(
        folder + 'wtf_im.jpg', cv_mine.LOAD_GRAY_SCALE)

    img = cv_mine.imReadAndConvert(
        folder + 'sans.jpg', cv_mine.LOAD_GRAY_SCALE)

    direction, magnitude, dx, dy = cv_mine.convDerivative(img)

    threshold = 0.2
    magnitude[magnitude > threshold] = 1
    magnitude[magnitude <= threshold] = 0

    # print(dst_mine[105:106, 103:106] - dst[100:103, 100:103])

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))
    plt.subplot(ax1), cv_mine.imgDisplay(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(ax2), cv_mine.imgDisplay(magnitude), plt.title('Magnitude')
    plt.xticks([]), plt.yticks([])
    plt.subplot(ax3), cv_mine.imgDisplay(dx), plt.title('dx')
    plt.xticks([]), plt.yticks([])
    plt.subplot(ax4), cv_mine.imgDisplay(direction), plt.title('direction')
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()


def showGausianBlur():
    img = cv_mine.imReadAndConvert(
        folder + 'wtf_im.jpg', cv_mine.LOAD_GRAY_SCALE)
    # print(cv_mine.blurImage2(img, 3))
    # print(cv_mine.blurImage1(img, 3))
    dst_cv = cv_mine.blurImage2(img, 13)
    dst_mine = cv_mine.blurImage1(img, 13)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3))
    plt.subplot(ax1), cv_mine.imgDisplay(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(ax2), cv_mine.imgDisplay(
        dst_mine), plt.title('Blue using mine')
    plt.xticks([]), plt.yticks([])
    plt.subplot(ax3), cv_mine.imgDisplay(dst_cv), plt.title('Blur using cv2')
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()


def showSobleEdgeDetection():
    img = cv_mine.imReadAndConvert(
        folder + 'wtf_im.jpg', cv_mine.LOAD_GRAY_SCALE)

    cv2_edges, my_edges = cv_mine.edgeDetectionSobel(img)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3))
    plt.subplot(ax1), cv_mine.imgDisplay(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(ax2), cv_mine.imgDisplay(
        my_edges), plt.title('Edges soble mine')
    plt.xticks([]), plt.yticks([])
    plt.subplot(ax3), cv_mine.imgDisplay(
        cv2_edges), plt.title('Edges soble cv2')
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()


def showLaplacianEdgeDetection():
    # img = cv_mine1.imReadAndConvert(
    #     'images/wtf_im.jpg', cv_mine1.LOAD_GRAY_SCALE)
    img = cv_mine.imReadAndConvert(
        folder + 'ghost.png', cv_mine.LOAD_GRAY_SCALE)
    img2 = cv_mine.imReadAndConvert(
        folder + 'boxman.jpg', cv_mine.LOAD_GRAY_SCALE)

    # cv2_edges, my_edges = cv_mine.edgeDetectionSobel(img)
    edges = cv_mine.edgeDetectionZeroCrossingSimple(img)
    edges_log = cv_mine.edgeDetectionZeroCrossingLOG(img2)

    images = [img, edges, img2, edges_log]
    labels = ["original", "Laplacian", "original", "Laplacian Log"]

    plot_images(images, labels)


def showCannyEdgeDetector():
    img2 = cv_mine.imReadAndConvert(
        folder + 'skeletor2.jpg', cv_mine.LOAD_GRAY_SCALE)
    # img2 = cv_mine1.imReadAndConvert(
    #     'images/sans.jpg', cv_mine1.LOAD_GRAY_SCALE)
    canny_cv2, canny_mine2 = cv_mine.edgeDetectionCanny(img2, 0.2, 0.1)
    images = [img2, canny_cv2, canny_mine2]
    labels = ["original", "canny cv", "Canny mine"]

    plot_images(images, labels)


def showHoughCircles():
    img2 = cv_mine.imReadAndConvert(
        folder + 'step_system.png', cv_mine.LOAD_GRAY_SCALE)
    img = cv_mine.imReadAndConvert(
        folder + 'step_system.png', cv_mine.LOAD_RGB)
    # img2 = cv_mine1.imReadAndConvert(
    #     'images/sans.jpg', cv_mine1.LOAD_GRAY_SCALE)
    # canny_cv2, canny_mine2 = cv_mine.edgeDetectionCanny(img2, 0.5, 0.35)
    # images = [img2, canny_cv2, canny_mine2]
    # labels = ["original", "canny cv", "Canny mine", "canny test"]

    circles = cv_mine.houghCircle(img2, 5, 80)

    ax = plt.gca()
    cv_mine.imgDisplay(img)
    for x, y, r in circles:
        circle = plt.Circle((x, y), r, color='r', fill=False)
        ax.add_patch(circle)
    plt.tight_layout()
    plt.show()


def main():
    print("My ID : 323363838")
    Show1DConv()
    Show2DConv()
    showDirEdges()
    showGausianBlur()
    showSobleEdgeDetection()
    showLaplacianEdgeDetection()

    showCannyEdgeDetector()
    showHoughCircles()


if __name__ == '__main__':
    main()
