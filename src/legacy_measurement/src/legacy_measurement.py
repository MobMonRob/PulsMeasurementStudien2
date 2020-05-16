#!/usr/bin/env python
from __future__ import print_function
from scipy import signal
from face_detector import FaceDetector
from pulse_publisher import PulsePublisher

import sys
import numpy as np
import time
import rospy
import matplotlib.pyplot as plt


class LegacyMeasurement(object):

    def __init__(self, is_video):
        self.is_video = is_video
        self.fps = 0
        self.buffer_size = 250
        self.data_buffer = []
        self.times = []
        self.ttimes = []
        self.samples = []
        self.freqs = []
        self.fft = []
        self.slices = [[0]]
        self.bpms = []
        self.bpm = 0
        self.MAX_BPM = 150
        self.MIN_BPM = 40
        self.pulse_sequence = 0
        self.publisher = PulsePublisher("legacy_measurement")
        self.publish_count = 0

    def on_image_frame(self, roi, timestamp):
        self.publish_count += 1
        self.times.append(timestamp.to_sec())

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

        # calculate heart rate every 30 frames
        if L == self.buffer_size and self.publish_count % 30 == 0:
            # remove linear trend on processed data to avoid interference of light change
            processed = signal.detrend(processed)

            # calculate fps
            self.fps = float(L) / (self.times[-1] - self.times[0])

            if self.is_video:
                interpolated = processed
            else:
                # calculate equidistant frame times
                even_times = np.linspace(self.times[0], self.times[-1], L)
                # interpolate the values for the even times
                interpolated = np.interp(x=even_times, xp=self.times, fp=processed)

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
                self.publisher.publish(self.bpm, timestamp)
                rospy.loginfo("[LegacyMeasurement] BPM: " + str(self.bpm))

        self.samples = processed

        # plot fourrier transform
        if L == self.buffer_size and self.publish_count % 30 == 0:
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

    def extractGreenColorChannel(self, frame):
        return frame[:, :, 1]


def main():
    rospy.init_node("legacy_measurement", anonymous=False, log_level=rospy.DEBUG)

    # Get ROS topic from launch parameter
    topic = rospy.get_param("~topic", "/webcam/image_raw")
    rospy.loginfo("[LegacyMeasurement] Listening on topic '" + topic + "'")

    video_file = rospy.get_param("~video_file", None)
    rospy.loginfo("[LegacyMeasurement] Video file input: '" + str(video_file) + "'")

    bdf_file = rospy.get_param("~bdf_file", "")
    rospy.loginfo("[LegacyMeasurement] Bdf file: '" + str(bdf_file) + "'")

    cascade_file = rospy.get_param("~cascade_file", "")
    rospy.loginfo("[LegacyMeasurement] Cascade file: '" + str(cascade_file) + "'")

    show_image_frame = rospy.get_param("~show_image_frame", False)
    rospy.loginfo("[LegacyMeasurement] Show image frame: '" + str(show_image_frame) + "'")

    # Start heart rate measurement
    is_video = video_file != ""
    pulse_measurement = LegacyMeasurement(is_video)

    face_detector = FaceDetector(topic, cascade_file)
    face_detector.bottom_face_callback = pulse_measurement.on_image_frame
    face_detector.run(video_file, bdf_file, show_image_frame)

    rospy.spin()
    rospy.loginfo("[LegacyMeasurement] Shutting down")


if __name__ == '__main__':
    sys.argv = rospy.myargv()
    main()
