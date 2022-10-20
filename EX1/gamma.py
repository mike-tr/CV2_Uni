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
from numpy.core.fromnumeric import shape
from numpy.random import gamma
from ex1_utils import LOAD_GRAY_SCALE
import ex1_utils as utils
import cv2
import numpy as np


def gammaDisplay(filename: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """

    try:
        alpha_slider_max = 200
        title_window = 'Gamma correction'

        # if gray scale use the util function else use CV2 and scale to 0-1
        if rep == LOAD_GRAY_SCALE:
            image = utils.imReadAndConvert(filename, LOAD_GRAY_SCALE)
        else:
            image: np.ndarray = cv2.imread(filename) / 255
    except:
        print("something went wrong")
        return

    if image is None:
        return

    def on_trackbar(val):
        # do the math pixel intensity in pow of gamma.
        gamma = (val / alpha_slider_max) * 2 + 0.01
        imgamma = image ** gamma
        print(image)

        cv2.imshow(title_window, imgamma)

    cv2.namedWindow(title_window)
    trackbar_name = "gamma "
    cv2.createTrackbar(trackbar_name, title_window, 100,
                       alpha_slider_max, on_trackbar)
    # Show some stuff
    on_trackbar(100)
    # Wait until user press some key
    cv2.waitKey()


def main():
    gammaDisplay('forest_mist.jpg', 2)


if __name__ == '__main__':
    main()
