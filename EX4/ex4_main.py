# ps2
import os
import numpy as np
from ex4_utils import *
import cv2


def displayDepthImage(l_img, r_img, disparity_range=(0, 5), method=disparitySSD):
    p_size = 2
    d_ssd = method(l_img, r_img, disparity_range, p_size)
    plt.matshow(d_ssd)
    plt.colorbar()
    plt.show()


def main():
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    print("ID:", myID())
    # 1-a
    # Read images
    i = 0
    folder = ''
    L = cv2.imread(os.path.join(folder, 'pair%d-L.png' % i), 0) / 255.0
    R = cv2.imread(os.path.join(folder, 'pair%d-R.png' % i), 0) / 255.0

    # L = cv2.imread(os.path.join(folder, 'jupleft.png'), 0) / 255.0
    # R = cv2.imread(os.path.join(folder, 'jupright.png'), 0) / 255.0
    # Display depth SSD
    displayDepthImage(L, R, (0, 6), method=disparitySSD)
    # Display depth NC
    displayDepthImage(L, R, (0, 6), method=disparityNC)

    i = 1
    L = cv2.imread(os.path.join(folder, 'pair%d-L.png' % i), 0) / 255.0
    R = cv2.imread(os.path.join(folder, 'pair%d-R.png' % i), 0) / 255.0

    # L = cv2.imread(os.path.join(folder, 'jupleft.png'), 0) / 255.0
    # R = cv2.imread(os.path.join(folder, 'jupright.png'), 0) / 255.0

    # # Display depth SSD
    # displayDepthImage(L, R, (5, 60), method=disparitySSD)
    # # Display depth NC
    # displayDepthImage(L, R, (5, 60), method=disparityNC)

    src = np.array([[279, 552],
                    [372, 559],
                    [362, 472],
                    [277, 469]])
    dst = np.array([[24, 566],
                    [114, 552],
                    [106, 474],
                    [19, 481]])
    h, e = computeHomography(src, dst)
    # h_src = Homogeneous(src)
    # pred = h.dot(h_src.T).T
    #
    # pred = unHomogeneous(pred)
    # print(np.sqrt(np.square(pred-dst).mean()))

    dst = cv2.cvtColor(cv2.imread(folder + 'earth.jpg'),
                       cv2.COLOR_BGR2RGB) / 255
    src = cv2.cvtColor(cv2.imread(folder + 'babun.jpg'),
                       cv2.COLOR_BGR2RGB) / 255
    mask = cv2.cvtColor(cv2.imread(folder + 'babun_mask.jpg'),
                        cv2.COLOR_BGR2RGB) / 255

    warpImagAdvanced(src, dst, mask)


if __name__ == '__main__':
    main()
