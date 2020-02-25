import numpy as np
import time
import cv2
import threading
import timeit
import wx

import pyfftw.interfaces.cache
from scipy import signal

import matplotlib.pyplot as plt


class PulseMeasurement:

    def __init__(self):

        self.roi = np.zeros((10, 10))

    def run(self, roi):
        self.roi = cv2.resize(roi, (200, 200))
        downsampledImage = cv2.pyrDown(self.roi)

        cv2.imshow("down", downsampledImage)
        cv2.waitKey(3)