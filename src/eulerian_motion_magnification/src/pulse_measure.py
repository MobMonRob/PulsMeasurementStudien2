#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import print_function
import numpy as np
import time
import cv2
import os
import rospy
import sys

from std_msgs.msg import Float32
import scipy.fftpack as fftpack
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
from face_detection import FaceDetector


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
    #npa[:, :, :, 0] = np.multiply(npa[:, :, :, 0], amplify)
    npa[:, :, :, 1] = np.multiply(npa[:, :, :, 1], amplify)
    #npa[:, :, :, 2] = np.multiply(npa[:, :, :, 2], amplify)
    return npa

# go back with the gaussian pyramids to retain higher resolution image
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

# show many images after each other (show many images in buffer)
def show_video(final):
    for image in final:
        time.sleep(0.15)
        cv2.imshow("final", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# showing images if buffer is similar to queue scenario
def show_image(final):
    image = final[-1]
    cv2.imshow("final", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pass


def calculate_pulse(upsampled_final_amplified, recorded_time):
    green_values = []
    for i in range(0, upsampled_final_amplified.shape[0]):
        img = upsampled_final_amplified[i]
        green_intensity = img[100][100][1]
        green_values.append(green_intensity)
    peaks, _ = find_peaks(green_values)
    print("peaks: " + str(peaks))
    pulse = (len(peaks) / float(recorded_time)) * 60
    pulse = np.int16(pulse)
    print("pulse :" + str(pulse))
    return pulse


class PulseMeasurement:

    def __init__(self):
        self.count = 0
        self.levels = 2
        self.low = 0.6
        self.high = 1.0
        self.amplification = 300
        self.pub_pulse = rospy.Publisher('eulerian_color_changes_pulse', Float32, queue_size=10)
        self.fps = 30
        self.video_array = []
        self.buffer_size = 0
        self.time_array = []
        self.calculating_at = 100
        self.recording_time = 10

    def calculate_fps(self):
        time_difference = self.time_array[-1] - self.time_array[0]
        if time_difference == 0:
            pass
        samplerate = self.buffer_size / time_difference
        print(samplerate)
        return samplerate

    def publish_pulse(self, pulse):
        msg_to_publish = pulse

        self.pub_pulse.publish(msg_to_publish)

    # calculate pulse after certain amount of images taken, calculation based on a larger amount of time
    def start_calulation(self, roi):
        # append timestamp to array
        self.time_array.append(time.time())
        # normalize, resize and downsample image
        normalized = cv2.normalize(roi.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        cropped = cv2.resize(normalized, (200, 200))
        gaussian_frame = build_gaussian_frame(cropped, self.levels)
        self.video_array.append(gaussian_frame)
        # check if recording images took longer than certain amount of time
        if (self.time_array[-1] - self.time_array[0]) >= self.recording_time:
            # determine how many pictures got buffered during time interval
            self.buffer_size = (len(self.time_array))
            # release first image and timestamp
            self.video_array.pop(0)
            self.time_array.pop(0)
            self.calculating_at = self.calculating_at + 1
            # calculate again after certain amount of images
            if self.calculating_at >= 50:
                print(len(self.video_array))
                self.calculate_fps()
                t = np.asarray(self.video_array, dtype=np.float32)
                filtered_tensor = temporal_ideal_filter(t, self.low, self.high, self.fps)
                amplified_video = amplify_video(filtered_tensor, amplify=self.amplification)
                # upsampled_final_t = upsample_final_video(t, self.levels)
                upsampled_final_amplified = upsample_final_video(amplified_video, self.levels)
                pulse = calculate_pulse(upsampled_final_amplified, self.recording_time)
                self.publish_pulse(pulse)
                # final = reconstruct_video(upsampled_final_amplified, upsampled_final_t, levels=3)
                # show_video(final)
                self.calculating_at = 0


def main():
    rospy.init_node('face_detection', anonymous=True, log_level=rospy.DEBUG)

    # Get ROS topic from launch parameter
    topic = rospy.get_param("~topic", "/webcam/image_raw")
    rospy.loginfo("Listening on topic '" + topic + "'")

    bdf_file = rospy.get_param("~bdf_file", "")
    rospy.loginfo("Bdf file: '" + str(bdf_file) + "'")

    cascade_file = rospy.get_param("~cascade_file", "")
    rospy.loginfo("Cascade file: '" + str(cascade_file) + "'")

    show_image_frame = rospy.get_param("~show_image_frame", False)
    rospy.loginfo("Show image frame: '" + str(show_image_frame) + "'")

    pulse_processor = PulseMeasurement()

    face_detector = FaceDetector(topic, cascade_file, show_image_frame)
    face_detector.face_callback = pulse_processor.start_calulation
    face_detector.run(bdf_file)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.argv = rospy.myargv()
    main()