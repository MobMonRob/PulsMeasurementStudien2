from scipy import signal
from pulse_chest_strap.msg import Pulse

import numpy as np
import time
import cv2
import rospy
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
        self.pulse_sequence = 0
        self.pulse_publisher = rospy.Publisher("/legacy_measurement/pulse", Pulse, queue_size=10)
        self.count = 0

    def extractGreenColorChannel(self, frame):
        return frame[:, :, 1]

    def run(self, roi, timestamp):
        self.count += 1
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
        if L == self.buffer_size and self.count % 30 == 0:
            # calculate fps
            # self.fps = float(L) / (self.times[-1] - self.times[0])
            self.fps = 61

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
                self.publish_pulse(self.bpm, timestamp)
                rospy.loginfo("BPM: " + str(self.bpm))

        self.samples = processed

        # plot fourrier transform
        if L == self.buffer_size and self.count % 30 == 0:
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

    def publish_pulse(self, pulse, timestamp):
        ros_msg = Pulse()
        ros_msg.pulse = pulse
        ros_msg.time.stamp = timestamp
        ros_msg.time.seq = self.pulse_sequence

        self.pulse_publisher.publish(ros_msg)
        self.pulse_sequence += 1
