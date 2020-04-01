import numpy as np
import time
import cv2
from scipy import signal

import matplotlib.pyplot as plt


class PulseMeasurement(object):

    def __init__(self, buffer_size=250):
        self.roi = np.zeros((10, 10))
        self.fps = 0
        self.buffer_size = 250
        self.data_buffer = []
        self.times = []
        self.ttimes = []
        self.samples = []
        self.freqs = []
        self.fft = []
        self.slices = [[0]]
        self.t0 = time.time()
        self.bpms = []
        self.bpm = 0
        self.MAX_BPM = 150
        self.MIN_BPM = 40

    def extractGreenColorChannel(self, frame):
        return frame[:, :, 1]

    def run(self, roi):
        self.times.append(time.time() - self.t0)
        self.roi = roi
        self.gray = cv2.equalizeHist(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))

        # calculate mean green from roi
        green_mean = np.mean(self.extractGreenColorChannel(roi))
        self.data_buffer.append(green_mean)

        # get number of frames processed
        L = len(self.data_buffer)

        # remove sudden changes, if the avg value change is over 10, use the previous green mean instead
        if abs(green_mean - np.mean(self.data_buffer)) > 10 and L > 99:
            self.data_buffer[-1] = self.data_buffer[-2]

        # only use a max amount of frames. Determined by buffer_size
        if L > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            L = self.buffer_size

        # create array from average green values of all processed frames
        processed = np.array(self.data_buffer)

        # start heart rate measurment after 10 frames
        if L == self.buffer_size:
            # calculate fps
            # self.fps = float(L) / (self.times[-1] - self.times[0])
            self.fps = 5

            # calculate equidistant frame times
            # even_times = np.linspace(self.times[0], self.times[-1], L)

            # remove linear trend on processed data to avoid interference of light change
            processed = signal.detrend(processed)

            # interpolate the values for the even times
            # interpolated = np.interp(x=even_times, xp=self.times, fp=processed)
            interpolated = processed

            # apply hamming window to make the signal become more periodic
            interpolated = np.hamming(L) * interpolated

            # normalize the interpolation
            norm = interpolated / np.linalg.norm(interpolated)

            # do a fast fourier transformation on the (real) interpolated values
            raw = np.fft.rfft(norm)

            # get amplitude spectrum
            self.fft = np.abs(raw) ** 2

            # create a list for mapping the fft frequencies to the correct bpm
            self.freqs = (float(self.fps) / L) * np.arange(L / 2 + 1)
            freqs = 60. * self.freqs

            # find indeces where the frequencey is within the expected heart rate range
            idx = np.where((freqs > self.MIN_BPM) & (freqs < self.MAX_BPM))

            # reduce fft to "interesting" frequencies
            self.fft = self.fft[idx]

            # reduce frequency list to "interesting" frequencies
            self.freqs = freqs[idx]

            # find the frequency with the highest amplitude
            if len(self.fft) > 0:
                idx2 = np.argmax(self.fft)
                self.bpm = self.freqs[idx2]
                self.bpms.append(self.bpm)
                self.visualize_heart_rate(raw, idx, idx2)

        self.samples = processed

        # visualize data
        # if L == self.buffer_size:
        #     green_mean_visualized = np.zeros((100,100,3))
        #     green_mean_visualized[:,:,2] += (self.data_buffer[-1] - np.mean(self.data_buffer))
        #     cv2.imshow('test',green_mean_visualized)

        # plot fourrier transform
        if L == self.buffer_size:
            index = np.arange(len(self.data_buffer))

            data = self.data_buffer - np.mean(self.data_buffer)

            plt.clf()
            plt.subplot(2, 1, 1)
            plt.plot(index, data, '.-')
            plt.title('Green value over time')
            plt.ylabel('Green value')
            plt.xlabel('last x frames')

            index = np.arange(len(self.freqs))
            plt.subplot(2, 1, 2)
            plt.bar(index, self.fft)
            plt.xlabel('Frequencies (bpm)', fontsize=10)
            plt.ylabel('Amplitude', fontsize=10)
            plt.xticks(index, [round(x, 2) for x in self.freqs], fontsize=10, rotation=30)
            plt.title('Fourier Transformation')
            plt.draw()
            plt.pause(0.001)

        return self.roi

    def visualize_heart_rate(self, raw, idx, idx2):
        phase = np.angle(raw)
        phase = phase[idx]

        t = (np.sin(phase[idx2]) + 1.) / 2.
        t = 0.9 * t + 0.1
        alpha = t
        beta = 1 - t

        r = alpha * self.roi[:, :, 0]
        g = alpha * self.roi[:, :, 1] + \
            beta * self.gray
        b = alpha * self.roi[:, :, 2]
        self.roi = cv2.merge([r, g, b])
        self.slices = [np.copy(self.roi[:, :, 1])]

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.lfilter(b, a, data)
        return y

    def reset(self):
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.samples = []
        self.times = []
        self.data_buffer = []
        self.fps = 0
        self.fft = []
        self.freqs = []
        self.t0 = time.time()
        self.bpm = 0
        self.bpms = []
