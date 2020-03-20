import collections
import numpy as np
import time
import cv2
import os

import scipy.fftpack as fftpack


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


def temporal_ideal_filter(gau_video, low, high, fps, axis=0):
    fft = fftpack.fft(gau_video, axis=axis)
    frequencies = fftpack.fftfreq(gau_video.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - low)).argmin()
    bound_high = (np.abs(frequencies - high)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low] = 0
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
        image = image + original_video[i]
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
        self.levels = 3
        self.low = 0.6
        self.high = 1.5
        self.amplification = 20

        self.fps = 10
        self.video_array = []
        self.start_time = 0
        self.end_time = 0
        self.butter_size = 20

    def calculate_fps(self):
        time_difference = self.end_time - self.start_time
        samplerate = self.butter_size / time_difference
        print(samplerate)
        return samplerate

    def run(self, roi):

        if self.count < self.butter_size:
            frame = roi
            normalized = cv2.normalize(frame.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
            cropped = cv2.resize(normalized, (200, 200))
            gaussian_frame = build_gaussian_frame(cropped, self.levels)
            self.video_array.append(gaussian_frame)
            self.count += 1
            if self.count == self.butter_size:
                self.end_time = time.time()
                if(self.start_time != 0):
                    self.fps = self.calculate_fps()
                else:
                    self.fps = 23
                print('reached 100 frames')
                t = np.asarray(self.video_array, dtype=np.float32)
                print('converted as np array')
                filtered_tensor = temporal_ideal_filter(t, self.low, self.high, self.fps)
                print('applied filter')
                print(np.asarray(filtered_tensor, dtype=np.float32).shape)
                amplified_video = amplify_video(filtered_tensor, amplify=self.amplification)
                print('amplified')
                print(np.asarray(amplified_video, dtype=np.float32).shape)
                upsampled_final_t = upsample_final_video(t, self.levels)
                print('upsampled original')
                print(np.asarray(upsampled_final_t, dtype=np.float32).shape)
                upsampled_final_amplified = upsample_final_video(amplified_video, self.levels)
                print('upsampled amplified')
                print(np.asarray(upsampled_final_amplified, dtype=np.float32).shape)
                final = reconstruct_vido(upsampled_final_amplified, upsampled_final_t, levels=3)
                print('reconstruted')
                show_video(final)
                self.count = 0
                self.video_array = []
                self.start_time = time.time()
        print(self.count)
