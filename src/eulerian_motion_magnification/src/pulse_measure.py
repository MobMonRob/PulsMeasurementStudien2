import numpy as np
import time
import cv2
from scipy import signal

import matplotlib.pyplot as plt

class PulseMeasurement(object):

    def __init__(self, buffer_size=250):
        self.roi = np.zeros((10, 10))

    def run(self, roi):
        self.roi = roi