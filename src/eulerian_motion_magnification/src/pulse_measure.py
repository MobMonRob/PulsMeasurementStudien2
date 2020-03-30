import collections
import numpy as np
import time
import cv2
import os
import rospy
from std_msgs.msg import Float32
import scipy.fftpack as fftpack
from scipy.signal import butter, filtfilt, find_peaks
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


def temporal_ideal_filter(gau_video, lowcut, highcut, fps, axis=0):
    fft = fftpack.fft(gau_video, axis=axis)
    frequencies = fftpack.fftfreq(gau_video.shape[0], d=1.0 / fps)
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


def reconstruct_video(amplified_video, original_video, levels=3):
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


def show_image(final):
    image = final[-1]
    cv2.imshow("final", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pass


def calculate_pulse(upsampled_final_amplified):
    green_values = []
    for i in range(0, upsampled_final_amplified.shape[0]):
        img = upsampled_final_amplified[i]
        green_intensity = img[100][100][1]
        green_values.append(green_intensity)
    peaks, _ = find_peaks(green_values)
    pulse = len(peaks) * 2
    pulse = np.int16(pulse)
    print("pulse :" + str(pulse))


class PulseMeasurement:

    def __init__(self):
        self.count = 0
        self.levels = 2
        self.low = 0.8
        self.high = 2
        self.amplification = 20
        self.pub_first = rospy.Publisher('eulerian_color_changes_first', Float32, queue_size=10)
        self.pub_second = rospy.Publisher('eulerian_color_changes_second', Float32, queue_size=10)
        self.pub_third = rospy.Publisher('eulerian_color_changes_third', Float32, queue_size=10)
        self.fps = 23
        self.video_array = []
        self.start_time = 0
        self.end_time = 0
        self.buffer_size = 0
        self.time_array = []
        self.calculating_at = 50

    def calculate_fps(self):
        time_difference = self.time_array[-1] - self.time_array[0]
        if time_difference == 0:
            pass
        samplerate = self.buffer_size / time_difference
        print(samplerate)
        return samplerate

    def publish_color_changes(self, upsampled_final_amplified):
        image = upsampled_final_amplified[-1, :, :, :]

        first_intensity = image[100][80][1]
        msg_to_publish_first = first_intensity

        second_intensity = image[100][100][1]
        msg_to_publish_second = second_intensity

        third_intensity = image[100][120][1]
        msg_to_publish_third = third_intensity

        self.pub_first.publish(msg_to_publish_first)
        self.pub_second.publish(msg_to_publish_second)
        self.pub_third.publish(msg_to_publish_third)

    def run(self, roi):
        self.time_array.append(time.time())
        normalized = cv2.normalize(roi.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        cropped = cv2.resize(normalized, (200, 200))
        gaussian_frame = build_gaussian_frame(cropped, self.levels)
        self.video_array.append(gaussian_frame)
        if (self.time_array[-1] - self.time_array[0]) >= 30:
            self.buffer_size = (len(self.time_array))
            self.video_array.pop(0)
            self.time_array.pop(0)
            self.calculating_at = self.calculating_at + 1
            if self.calculating_at >= 50:
                self.calculate_fps()
                t = np.asarray(self.video_array, dtype=np.float32)
                filtered_tensor = temporal_ideal_filter(t, self.low, self.high, self.fps)
                amplified_video = amplify_video(filtered_tensor, amplify=self.amplification)
                # upsampled_final_t = upsample_final_video(t, self.levels)
                upsampled_final_amplified = upsample_final_video(amplified_video, self.levels)
                calculate_pulse(upsampled_final_amplified)
                # self.publish_color_changes(upsampled_final_amplified)
                # final = reconstruct_video(upsampled_final_amplified, upsampled_final_t, levels=3)
                # show_video(final)
                self.calculating_at = 0
