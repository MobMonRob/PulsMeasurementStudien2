#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
import time
import cv2
import rospy
import sys

import scipy.fftpack as fftpack
from scipy.signal import find_peaks
from face_detector import FaceDetector
from pulse_publisher import PulsePublisher


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
    :param normalized_frame: frame to downsample
    :param level: how deep gaussian pyramid needs be calculated
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
    amplification_array = amplification_array + filtered_video
    return amplification_array


def calculate_pulse(processed_video, recorded_time, show_processed_image):
    """
    The processed images, saved in processed_video, is used as the data basis to calculate the pulse.
    For the actual calculation, only the red values are needed. Therefore these are extracted with the method
    extract_red_values.
    The mean value of all the pixels in an image is calculated and added to list red_values.
    On this array the peaks are detected and the amount of peaks is used to calculate bpm.
    : param recorded_time: Timespan, where images are collected in array
    """
    if show_processed_image:
        processed_video = extract_red_values(processed_video, show_processed_image)
        processed_video = np.asarray(processed_video, dtype=np.float32)
    red_values = []
    for i in range(0, processed_video.shape[0]):
        img = processed_video[i]
        red_intensity = np.mean(img)
        red_values.append(red_intensity)
    peaks, _ = find_peaks(red_values)
    pulse = (len(peaks) / float(recorded_time)) * 60
    pulse = np.int16(pulse)
    rospy.loginfo("[EulerianMotionMagnification] Pulse: " + str(pulse))
    return pulse, red_values


def extract_red_values(gaussian_frame, show_processed_image):
    """
    Filters red-intensities of the image (Color channel 2) and adds to red_values list
    """
    if show_processed_image:
        red_values = []
        for i in range(0, gaussian_frame.shape[0]):
            red_value = gaussian_frame[i, :, 2]
            red_values.append(red_value)
    else:
        red_values = gaussian_frame[:, :, 2]
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

    def __init__(self, show_processed_image):
        self.show_processed_image = show_processed_image
        self.count = 0
        self.levels = 2
        self.low = 0.9
        self.high = 1.7
        self.amplification = 30
        self.publisher = PulsePublisher("eulerian_motion_magnification")
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
        rospy.loginfo("[EulerianMotionMagnification] FPS: " + str(self.fps))
        rospy.loginfo("[EulerianMotionMagnification] Video array length: " + str(len(self.video_array)))

    # calculate pulse after certain amount of images taken, calculation based on a larger amount of time
    def start_calulation(self, roi, timestamp):
        # append timestamp to array
        self.time_array.append(timestamp)
        # normalize, resize and downsample image
        normalized = cv2.normalize(roi.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        cropped = cv2.resize(normalized, (200, 200))
        gaussian_frame = build_gaussian_frame(cropped, self.levels)
        if self.show_processed_image:
            self.video_array.append(gaussian_frame)
        else:
            red_values_images = extract_red_values(gaussian_frame, self.show_processed_image)
            self.video_array.append(red_values_images)
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
                rospy.loginfo("[EulerianMotionMagnification] Length final " + str(len(self.video_array)))
                self.calculate_fps()
                copy_video_array = np.copy(self.video_array)
                copy_video_array = np.asarray(copy_video_array, dtype=np.float32)
                copy_video_array = temporal_bandpass_filter(copy_video_array, self.low, self.high, self.fps)
                if self.show_processed_image:
                    self.arrayLength = len(self.video_array)
                    processed_video = amplify_video(copy_video_array, amplify=self.amplification)
                    show_images(processed_video, self.video_array, self.arrayLength, self.isFirst, self.levels, self.calculating_boarder, self.fps)
                else:
                    copy_video_array = amplify_video(copy_video_array, amplify=self.amplification)
                pulse, red_values = calculate_pulse(copy_video_array, self.recording_time, self.show_processed_image)
                self.publisher.publish(pulse, timestamp)
                self.calculating_at = 0
                self.isFirst = False


def main():
    rospy.init_node('face_detection', anonymous=True, log_level=rospy.DEBUG)

    # Get ROS topic from launch parameter
    topic = rospy.get_param("~topic", "/webcam/image_raw")
    rospy.loginfo("[EulerianMotionMagnification] Listening on topic '" + topic + "'")

    video_file = rospy.get_param("~video_file", None)
    rospy.loginfo("[EulerianMotionMagnification] Video file input: '" + str(video_file) + "'")

    bdf_file = rospy.get_param("~bdf_file", "")
    rospy.loginfo("[EulerianMotionMagnification] Bdf file: '" + str(bdf_file) + "'")

    cascade_file = rospy.get_param("~cascade_file", "")
    rospy.loginfo("[EulerianMotionMagnification] Cascade file: '" + str(cascade_file) + "'")

    show_image_frame = rospy.get_param("~show_image_frame", False)
    rospy.loginfo("[EulerianMotionMagnification] Show image frame: '" + str(show_image_frame) + "'")

    show_processed_image = rospy.get_param("~show_processed_image", False)
    rospy.loginfo("[EulerianMotionMagnification] Show processed frame: '" + str(show_processed_image) + "'")

    pulse_processor = PulseMeasurement(show_processed_image)

    face_detector = FaceDetector(topic, cascade_file)
    face_detector.face_callback = pulse_processor.start_calulation
    face_detector.run(video_file, bdf_file, show_image_frame)

    rospy.spin()
    rospy.loginfo("[EulerianMotionMagnification] Shutting down")

    # Destroy windows on close
    cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.argv = rospy.myargv()
    main()
