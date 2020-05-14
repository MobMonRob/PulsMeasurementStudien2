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
    """
    Logic to build gaussian pyramid. Each level is 1/4 of the last image that's passed in.
    The pyramid list contains all processed images, including the not processed original one.
    The whole pyramid is passed to the method build_gaussian_frame
    """
    s = frame.copy()
    pyramid = [s]
    for i in range(level):
        s = cv2.pyrDown(s)
        pyramid.append(s)
    return pyramid


def build_gaussian_frame(normalized_frame, level):
    """
    Build gaussian pyramid. Level is indicating how deep the gaussian pyramid has to be.
    The actual logic to downsample is in method build_gaussian_pyramid.
    Only the last frame of the pyramid is needed.
    :param level: how deep gaussian pyramid needs be calculated
    :param normalized_frame: frame to downsample
    """
    pyramid = build_gaussian_pyramid(normalized_frame, level)
    gaussian_frame = pyramid[-1]
    return gaussian_frame


def temporal_bandpass_filter(video_to_filter, low, high, fps):
    """
    Filters colour-intensity changes that conform to the low and high frequencies.
    Colour-intensity changes should between low and high frequencies.
    :param video_to_filter:
    :param low: frequencies for lower bound
    :param high: frequencies for higher bound
    """
    fft = fftpack.fft(video_to_filter, axis=0)
    frequencies = fftpack.fftfreq(video_to_filter.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - low)).argmin()
    bound_high = (np.abs(frequencies - high)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    iff = fftpack.ifft(fft, axis=0)
    return iff


def amplify_video(filtered_video, amplify):
    """
    Multiply the filtered colour-intensity-changes with the amplifying factor
    :param filtered_video: array, containing frames after passing temporal_bandpass_filter
    :param amplify: indicates by how much changes need to be amplified
    """
    amplification_array = np.asarray(filtered_video, dtype=np.float32)
    amplification_array = np.multiply(amplification_array, amplify)
    return amplification_array


def calculate_pulse(processed_video, recorded_time):
    """
    The processed images, saved in processed_video, is used as the data basis to calculate the pulse.
    For the actual calculation, only the red values are needed. Therefore these are extracted with the method
    extract_red_values.
    The mean value of all the pixels in an image is calculated and added to list red_values.
    On this array the peaks are detected and the amount of peaks is used to calculate bpm.
    : param recorded_time: Timespan, where images are collected in array
    """
    processed_video = extract_red_values(processed_video)
    processed_video = np.asarray(processed_video, dtype=np.float32)
    red_values = []
    for i in range(0, processed_video.shape[0]):
        img = processed_video[i]
        red_intensity = np.mean(img)
        red_values.append(red_intensity)
    peaks, _ = find_peaks(red_values)
    pulse = (len(peaks) / float(recorded_time)) * 60
    pulse = np.int16(pulse)
    print("pulse :" + str(pulse))
    return pulse, red_values


def extract_red_values(processed_video):
    """
    Filters red-intensities of the image (Color channel 2) and adds to red_values list
    """
    red_values = []
    for i in range(0, processed_video.shape[0]):
        red_value = processed_video[i, :, 2]
        red_values.append(red_value)
    return red_values


def upsample_images(processed_video, unprocessed_video, arraylength, levels):
    """
    Upsample images in video sequence to be able to display them.
    Iterate through each image and add processed image to unprocessed image.
    Upsample resulting image by reversing gaussian pyramide.
    """
    processed_video = np.asarray(processed_video, dtype=np.float32)
    upsampled_images = np.zeros((arraylength, 200, 200, 3))
    for i in range(0, processed_video.shape[0]):
        img = processed_video[i]
        img = img + unprocessed_video[i]
        for x in range(levels):
            img = cv2.pyrUp(img)
        upsampled_images[i] = img
    return upsampled_images


def show_images(processed_video, unprocessed_video, arraylength, isFirst, levels, calculating_boarder, fps):
    """
    Show upsampled images to make color-intensity-changes visible to the human eye.
    Show whole video-sequence if it's the first time executing. If it's not the first time,
    show only new images.
    :param processed_video: Array, containing processed images
    :param unprocessed_video: Array, containing original frames without filtering and amplification
    :param isFirst: indicates, whether first round of calculation
    :param levels: depth of gaussian pyramide, indicates how many rounds are needed to upsample
    :param calculating_boarder: how many images are new since last calculation
    """
    processed_video = upsample_images(processed_video, unprocessed_video, arraylength, levels)
    if not isFirst:
        processed_video = processed_video[-calculating_boarder:]
    for image in processed_video:
        time.sleep(1/fps)
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
        """
        calculate fps of incoming frames by calculating time-difference of the first timestamp (first image
        in array) and last timestamp (last image in array)
        """
        time_difference = self.time_array[-1] - self.time_array[0]
        time_difference_in_seconds = time_difference.to_sec()
        if time_difference_in_seconds == 0:
            pass
        self.fps = self.buffer_size / time_difference_in_seconds
        print("fps:" + str(self.fps))

    def publish_pulse(self, pulse, red_values):
        """
        Publish calculated pulse to ROS. Message is of type Float32.
        :param pulse: calculated pulse value
        """
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
                video_to_process = np.copy(self.video_array)
                video_to_process = np.asarray(video_to_process, dtype=np.float32)
                video_to_process = temporal_bandpass_filter(video_to_process, self.low, self.high, self.fps)
                processed_video = amplify_video(video_to_process, amplify=self.amplification)
                pulse, red_values = calculate_pulse(video_to_process, self.recording_time)
                self.publish_pulse(pulse, red_values)
                show_images(processed_video, self.video_array, self.arrayLength, self.isFirst, self.levels, self.calculating_boarder, self.fps)
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
