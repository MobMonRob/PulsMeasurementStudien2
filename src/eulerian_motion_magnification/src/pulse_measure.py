import collections
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
        self._maxHistoryLength = 30
        self._array1 = []
        self._array2 = []
        self._counter = 0

    def run(self, roi):
        #read video
        self.roi = cv2.resize(roi, (200, 200))


        #normalize and Level 2 Gaussian Blurr
        out = cv2.normalize(self.roi.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        downsamplelevel1 = cv2.pyrDown(out)
        downsamplelevel2 = cv2.pyrDown(downsamplelevel1)
        #cv2.imshow("normalized", downsamplelevel2)
        cv2.waitKey(3)
        # save in numpy array
        if self._counter < self._maxHistoryLength:
            self._array1.append(downsamplelevel2)
            cv2.imshow("stack", self._array1[self._counter])
            self._counter = self._counter + 1
            return
        else:
            self._counter = 0
            self._array2 = self._array1
            self._array1 = []
            print('stop')
        #temporal filtering (ideal bandpassing)
        #amplify
        #render outputvideo