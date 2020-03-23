import collections
import numpy as np
import time
import cv2
import os

import scipy.fftpack as fftpack
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt


def build_gaussian_pyramid(frame, level=3):
    s = frame.copy()
    pyramid = [s]
    for i in range(level):
        s = cv2.pyrDown(s)
        pyramid.append(s)
    return pyramid


def build_gaussian_frame(normalized, level):
    pyramid = build_gaussian_pyramid(normalized, level)
    gaussian_frame = pyramid[-1]
    return gaussian_frame


def show_a_plot(function, phase):
    plot_array_blue = []
    plot_array_green = []
    plot_array_red = []
    x_axis = []
    for i in range(0, function.shape[0]):
        image = function[i, :, :, :]
        point_of_interest_blue = image[100][100][0]
        point_of_interest_green = image[100][100][1]
        point_of_interest_red = image[100][100][2]
        x_axis.append(i)
        plot_array_blue.append(point_of_interest_blue)
        plot_array_green.append(point_of_interest_green)
        plot_array_red.append(point_of_interest_red)
    plt.figure()
    plt.subplot(311)
    plt.ylabel('blue intensity')
    plt.ylim(0, 3)
    plt.plot(x_axis, plot_array_blue)

    plt.subplot(312)
    plt.ylabel('green intensity')
    plt.ylim(0, 3)
    plt.plot(x_axis, plot_array_green)

    plt.subplot(313)
    plt.ylabel('red intensity')
    plt.ylim(0, 3)
    plt.plot(x_axis, plot_array_red)
    plt.savefig('plot_' + str(phase) + '.png')


def temporal_ideal_filter(gau_video, lowcut, highcut, fps, axis=0):
    fft = fftpack.fft(gau_video, axis=axis)
    frequencies = fftpack.fftfreq(gau_video.shape[0], d=1.0 /fps)
    bound_low = (np.abs(frequencies - lowcut)).argmin()
    bound_high = (np.abs(frequencies - highcut)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    iff = fftpack.ifft(fft, axis=axis)
    return np.abs(iff)


def amplify_video(filtered_tensor, amplify):
    npa = np.asarray(filtered_tensor, dtype=np.float32)
    npa[:, :, :, 0] = np.multiply(npa[:, :, :, 0], amplify)
    npa[:, :, :, 1] = np.multiply(npa[:, :, :, 1], amplify)
    npa[:, :, :, 2] = np.multiply(npa[:, :, :, 2], amplify)
    return npa


def upsample_final_video(final, levels):
    final_video = []
    for i in range(0, final.shape[0]):
        img = final[i]
        for x in range(levels):
            img = cv2.pyrUp(img)
        final_video.append(img)
    return np.asarray(final_video, dtype=np.float32)


def reconstruct_vido(amplified_video, original_video, levels=3):
    final_video = np.zeros(original_video.shape)
    for i in range(0, amplified_video.shape[0]):
        image = amplified_video[i]
        image = original_video[i] + image
        final_video[i] = image
    return final_video


def show_video(final):
    for image in final:
        time.sleep(0.15)
        cv2.imshow("final", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


class PulseMeasurement:

    def __init__(self):
        self.count = 0
        self.levels = 2
        self.low = 0.8
        self.high = 1.8
        self.amplification = 20

        self.fps = 23
        self.video_array = []
        self.start_time = 0
        self.end_time = 0
        self.buffer_size = 100
        self.time_array = []
        self.phase = 0

    def calculate_fps(self):
        time_difference = self.time_array[-1] - self.time_array[0]
        samplerate = self.buffer_size / time_difference
        print(samplerate)
        return samplerate

    def run(self, roi):

        if self.count < self.buffer_size:
            self.time_array.append(time.time())
            frame = roi
            normalized = cv2.normalize(frame.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
            cropped = cv2.resize(normalized, (200, 200))
            gaussian_frame = build_gaussian_frame(cropped, self.levels)
            self.video_array.append(gaussian_frame)
            self.count += 1
            if self.count == self.buffer_size:
                self.phase +=1
                self.calculate_fps()
                print('reached required frames')
                t = np.asarray(self.video_array, dtype=np.float32)
                filtered_tensor = temporal_ideal_filter(t, self.low, self.high, self.fps)
                amplified_video = amplify_video(filtered_tensor, amplify=self.amplification)
                upsampled_final_t = upsample_final_video(t, self.levels)
                upsampled_final_amplified = upsample_final_video(amplified_video, self.levels)
                final = reconstruct_vido(upsampled_final_amplified, upsampled_final_t, levels=3)
                show_a_plot(final, self.phase)
                show_video(final)
                self.count = 0
                self.video_array = []
                self.time_array = []
