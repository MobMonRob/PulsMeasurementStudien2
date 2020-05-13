#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import print_function
import numpy as np
import time
import cv2
import os
import rospy
import sys
import thread

from std_msgs.msg import Float32, Float32MultiArray
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


def do_filtering_on_all(what_to_filter, low, high, fps):
    fft = fftpack.fft(what_to_filter, axis=0)
    frequencies = fftpack.fftfreq(what_to_filter.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - low)).argmin()
    bound_high = (np.abs(frequencies - high)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    iff = fftpack.ifft(fft, axis=0)
    return iff


def amplify_video(filtered_tensor, amplify):
    amplification_array = np.asarray(filtered_tensor, dtype=np.float32)
    amplification_array = np.multiply(amplification_array, amplify)
    return amplification_array


def calculate_pulse(amplified_filtered_frames, recorded_time):
    amplified_filtered_frames = extract_red_values(amplified_filtered_frames)
    amplified_filtered_frames = np.asarray(amplified_filtered_frames, dtype=np.float32)
    red_values = []
    for i in range(0, amplified_filtered_frames.shape[0]):
        img = amplified_filtered_frames[i]
        red_intensity = np.mean(img)
        red_values.append(red_intensity)
    peaks, _ = find_peaks(red_values)
    pulse = (len(peaks) / float(recorded_time)) * 60
    pulse = np.int16(pulse)
    print("pulse :" + str(pulse))
    return pulse, red_values


def extract_red_values(amplified_filtered_frames):
    red_values = []
    for i in range(0, amplified_filtered_frames.shape[0]):
        red_value = amplified_filtered_frames[i, :, 2]
        red_values.append(red_value)
    return red_values


def upsample_images(copy_video_array, video_array, arraylength, levels):
    copy_video_array = np.asarray(copy_video_array, dtype=np.float32)
    upsampled_images = np.zeros((arraylength, 200, 200, 3))
    for i in range(0, copy_video_array.shape[0]):
        img = copy_video_array[i]
        img = img + video_array[i]
        for x in range(levels):
            img = cv2.pyrUp(img)
        upsampled_images[i] = img
    return upsampled_images


def show_images(copy_video_array, video_array, arraylength, isFirst, levels, calculating_boarder):
    copy_video_array = upsample_images(copy_video_array, video_array, arraylength, levels)
    if not isFirst:
        copy_video_array = copy_video_array[-calculating_boarder:]
    for image in copy_video_array:
        time.sleep(0.0166)
        cv2.imshow("colour changes pulse", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


class PulseMeasurement:

    def __init__(self):
        self.count = 0
        self.levels = 2
        self.low = 1.3
        self.high = 1.7
        self.amplification = 30
        self.pub_pulse = rospy.Publisher('eulerian_color_changes_pulse', Float32, queue_size=10)
        self.fps = 30
        self.video_array = []
        self.buffer_size = 0
        self.time_array = []
        self.calculating_at = 0
        self.calculating_boarder = 50
        self.recording_time = 10
        self.isFirst = True
        self.arrayLength = 0

    def calculate_fps(self):
        time_difference = self.time_array[-1] - self.time_array[0]
        time_difference_in_seconds = time_difference.to_sec()
        if time_difference_in_seconds == 0:
            pass
        self.fps = self.buffer_size / time_difference_in_seconds
        print("fps:" + str(self.fps))
        print(len(self.video_array))

    def publish_pulse(self, pulse, red_values):
        msg_to_publish_pulse = pulse
        self.pub_pulse.publish(msg_to_publish_pulse)

    # calculate pulse after certain amount of images taken, calculation based on a larger amount of time
    def start_calulation(self, roi, timestamp):
        # append timestamp to array
        self.time_array.append(timestamp)
        # normalize, resize and downsample image
        normalized = cv2.normalize(roi.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        cropped = cv2.resize(normalized, (200, 200))
        gaussian_frame = build_gaussian_frame(cropped, self.levels)
        self.video_array.append(gaussian_frame)
        # check if recording images took longer than certain amount of time
        time_difference = self.time_array[-1] - self.time_array[0]
        time_difference_in_seconds = time_difference.to_sec()
        if time_difference_in_seconds >= self.recording_time:
            self.buffer_size = (len(self.time_array))
            # determine how many pictures got buffered during time interval
            # release first image and timestamp
            self.video_array.pop(0)
            self.time_array.pop(0)
            self.calculating_at = self.calculating_at + 1
            # calculate again after certain amount of images
            if self.calculating_at >= self.calculating_boarder:
                self.arrayLength = len(self.video_array)
                print("length final " + str(self.arrayLength))
                self.calculate_fps()
                copy_video_array = np.copy(self.video_array)
                copy_video_array = np.asarray(copy_video_array, dtype=np.float32)
                copy_video_array = do_filtering_on_all(copy_video_array, self.low, self.high, self.fps)
                final_array = amplify_video(copy_video_array, amplify=self.amplification)
                pulse, red_values = calculate_pulse(copy_video_array, self.recording_time)
                self.publish_pulse(pulse, red_values)
                show_images(final_array, self.video_array, self.arrayLength, self.isFirst, self.levels, self.calculating_boarder)
                self.calculating_at = 0
                self.isFirst = False


def main():
    rospy.init_node('face_detection', anonymous=True, log_level=rospy.DEBUG)

    # Get ROS topic from launch parameter
    topic = rospy.get_param("~topic", "/webcam/image_raw")
    rospy.loginfo("Listening on topic '" + topic + "'")

    video_file = rospy.get_param("~video_file", None)
    rospy.loginfo("Video file input: '" + str(video_file) + "'")

    bdf_file = rospy.get_param("~bdf_file", "")
    rospy.loginfo("Bdf file: '" + str(bdf_file) + "'")

    cascade_file = rospy.get_param("~cascade_file", "")
    rospy.loginfo("Cascade file: '" + str(cascade_file) + "'")

    show_image_frame = rospy.get_param("~show_image_frame", False)
    rospy.loginfo("Show image frame: '" + str(show_image_frame) + "'")

    pulse_processor = PulseMeasurement()

    face_detector = FaceDetector(topic, cascade_file)
    face_detector.face_callback = pulse_processor.start_calulation
    face_detector.run(video_file, bdf_file, show_image_frame)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.argv = rospy.myargv()
    main()
