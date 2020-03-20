#!/usr/bin/env python
# -*- encoding: utf-8 -*-

__version__ = "0.1.1"

import sys
import numpy as np
import cv2
import rospy
from pulse_chest_strap.msg import pulse
from scipy import interpolate
from scipy.signal import butter, lfilter, filtfilt, find_peaks
from cv_bridge import CvBridge, CvBridgeError
from face_detection.msg import Mask
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

class PulseHeadMovement:

    def __init__(self, topic):
        self.topic = topic
        self.bridge = CvBridge()
        # set up ROS publisher and node
        self.pub = rospy.Publisher('head_movement_pulse', pulse, queue_size=10)
        self.seq = 0
        self.lk_params = dict( winSize  = (35, 35),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.03))
        self.points_to_track = None
        self.prev_image = None
        self.track_len = 32
        self.refresh_rate = 500
        self.fps = 0
        self.frame_index = 0
        self.y_tracking_signal = None
        self.time_array = np.empty(self.refresh_rate-1)
        self.publish_time = None

    def run(self):
        rospy.Subscriber(self.topic, Mask, self.pulse_callback)
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down")

    def pulse_callback(self, mask):
        rospy.loginfo("Capture frame: " + str(self.frame_index))
        try:
            # Convert ROS image to OpenCV image
            original_image = self.bridge.imgmsg_to_cv2(mask.image)
            bottom_mask = self.bridge.imgmsg_to_cv2(mask.bottom_face_mask)
            forehead_mask = self.bridge.imgmsg_to_cv2(mask.forehead_mask)
            # self.show_image_with_mask(original_image,forehead_mask,bottom_mask)
            # refresh the points to track after a certain frame rate
            if self.frame_index % self.refresh_rate == 0 \
                    or self.points_to_track is None \
                    or len(self.points_to_track) < 5:
                if self.y_tracking_signal is not None:
                    # We already stored some y points from which we want to calculate the pulse
                    self.process_saved_points()
                if self.points_to_track is not None and len(self.points_to_track) < 5:
                    self.frame_index -= 1
                # get initial tracking points
                self.prev_image = original_image
                self.points_to_track = self.get_points_to_track(original_image, forehead_mask, bottom_mask)
                self.frame_index += 1
                return
            if self.frame_index % self.refresh_rate == self.refresh_rate-1:
                self.publish_time = mask.time.stamp
            if self.points_to_track is not None:
                self.calculate_optical_flow(original_image, mask.time.stamp)
            self.prev_image = original_image
            self.frame_index += 1

        except CvBridgeError as e:
            rospy.logerr(e)
            return

    def get_points_to_track(self, image, forehead_mask, bottom_mask):
        image = cv2.equalizeHist(image)
        # get the tracking points in the bottom face region
        # parameter for feature points in bottom face region. As there are some feature rich points, the quality level
        # is low to include more points
        bottom_feature_params = dict(maxCorners=100, qualityLevel=0.01, minDistance=7, blockSize=7)
        bottom_points = cv2.goodFeaturesToTrack(image, mask=bottom_mask, **bottom_feature_params)
        feature_points = np.array(bottom_points, dtype=np.float32)
        # get the tracking points in the forehead region
        # parameter for feature points in forehead region. As most points are not very feature rich, the quality level
        # is high to enable better tracking with lk tracker
        forehead_feature_params = dict(maxCorners=100, qualityLevel=0.01, minDistance=7, blockSize=7)
        forehead_points = cv2.goodFeaturesToTrack(image, mask=forehead_mask, **forehead_feature_params)
        forehead_points = np.array(forehead_points, dtype=np.float32)
        # put tracking points of both regions in one array and return feature points
        if feature_points.ndim == forehead_points.ndim:
            feature_points = np.append(feature_points, forehead_points, axis=0)
        elif feature_points.size > 0:
            pass
        elif forehead_points.size > 0:
            feature_points = forehead_points
        if len(feature_points) < 5:
            self.y_tracking_signal = None
        else:
            self.y_tracking_signal = np.empty([len(feature_points), self.refresh_rate-1], dtype=np.float32)
        return feature_points

    def calculate_optical_flow(self, image, time):
        # track points with lucas kanade tracker
        # make a copy for visualization
        vis = image.copy()
        if len(self.points_to_track) > 0:
            img0, img1 = cv2.equalizeHist(self.prev_image), cv2.equalizeHist(image)
            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, self.points_to_track, None, **self.lk_params)
            self.time_array[(self.frame_index%self.refresh_rate)-1] = time.to_sec()
            point_index = 0
            for p in p1:
                cv2.circle(vis, (p[0][0], p[0][1]), 2, (0, 255, 0), -1)
                self.y_tracking_signal[point_index][(self.frame_index%self.refresh_rate)-1] = p[0][1]
                point_index += 1

            self.points_to_track = p1
            cv2.imshow('lk_track', vis)
            cv2.waitKey(3)

    def process_saved_points(self):
        self.calculate_fps()
        rospy.loginfo("FPS: " + str(self.fps))
        interpolated_points = self.interpolate_points()
        filtered_signal = self.apply_butterworth_filter(interpolated_points)
        pca_array = self.process_PCA(filtered_signal)
        signal = self.find_most_periodic_signal(pca_array)
        pulse = self.calculate_pulse(signal)
        self.publish_pulse(pulse)
        return

    def calculate_fps(self):
        timespan = self.time_array[-1]-self.time_array[0]
        rospy.loginfo("Measured timespan: "+str(timespan))
        self.fps = self.refresh_rate/timespan

    def interpolate_points(self):
        sample_rate = 250
        stepsize = 1./sample_rate
        print(self.time_array[0])
        print(self.time_array[-1])
        xs = np.arange(self.time_array[0], self.time_array[-1],stepsize)
        interpolated_points = np.empty([np.size(self.y_tracking_signal,0), np.size(xs)])
        point_index = 0
        for row in self.y_tracking_signal:
            cs = interpolate.interp1d(self.time_array, row, kind="cubic",copy=False,axis=0)
            array_interpolated = cs(xs)
            interpolated_point_index = 0
            for point in array_interpolated:
                interpolated_points[point_index][interpolated_point_index] = point
                interpolated_point_index += 1
            point_index+=1
        #rospy.loginfo(len(interpolated_points[0])/(self.time_array[-1]-self.time_array[0]))
        # np.savetxt("/home/studienarbeit/Dokumente/y_points.csv", interpolated_points, delimiter=";")
        # plt.figure(figsize=(6.5, 4))
        # plt.plot(self.time_array, self.y_tracking_signal[8],label="points")
        # plt.figure(figsize=(6.5, 4))
        # plt.plot(xs, interpolated_points[8], label="S")
        # plt.show()
        return interpolated_points

    def apply_butterworth_filter(self, input_signal):
        sample_rate = len(input_signal[0])/(self.time_array[-1]-self.time_array[0])
        rospy.loginfo("sample rate: "+str(sample_rate))
        lowcut = 0.75
        highcut = 5
        filtered_signal = np.empty([np.size(input_signal, 0), np.size(input_signal, 1)])
        rospy.loginfo("rows:"+str(np.size(input_signal, 0)))
        point_index = 0
        for point in input_signal:
            filtered_points = butter_bandpass_filter(point, lowcut, highcut, sample_rate, order=5)
            filtered_point_index = 0
            for filtered_point in filtered_points:
                filtered_signal[point_index][filtered_point_index] = filtered_point
                filtered_point_index+=1
            point_index += 1
        # np.savetxt("/home/studienarbeit/Dokumente/y_points_filtered.csv", filtered_signal, delimiter=";")
        input_signal = None
        # stepsize = 1. / sample_rate
        # xs = np.arange(self.time_array[0], self.time_array[-1], stepsize)
        # plt.figure(figsize=(6.5, 4))
        # plt.plot(xs, filtered_signal[6], label="S")
        # plt.show()
        return filtered_signal

    def process_PCA(self, filtered_signal):
        sample_rate = len(filtered_signal[0]) / (self.time_array[-1] - self.time_array[0])
        filtered_signal = filtered_signal.transpose()
        pca = PCA(n_components=5)
        pca_array=pca.fit_transform(filtered_signal)
        pca_array = pca_array.transpose()
        # stepsize = 1. / sample_rate
        # xs = np.arange(self.time_array[0], self.time_array[-1], stepsize)
        # plt.figure(figsize=(6.5, 4))
        # for row in pca_array:
        #     plt.plot(xs, row, label="S")
        # plt.show()
        return pca_array

    def find_most_periodic_signal(self, pca_array):
        highest_correlation = 0
        best_signal = None
        for signal in pca_array:
            series = pd.Series(signal)
            autocorrelation = series.autocorr()
            #rospy.loginfo(autocorrelation)
            if autocorrelation > highest_correlation:
                best_signal = signal
        # sample_rate = len(pca_array[0]) / (self.time_array[-1] - self.time_array[0])
        # stepsize = 1. / sample_rate
        # pca_array = None
        # xs = np.arange(self.time_array[0], self.time_array[-1], stepsize)
        # plt.figure(figsize=(6.5, 4))
        # plt.plot(xs, best_signal, label="S")
        # plt.show()
        return best_signal

    def calculate_pulse(self, signal):
        # sample_rate = len(signal) / (self.time_array[-1] - self.time_array[0])
        # stepsize = 1. / sample_rate
        # xs = np.arange(self.time_array[0], self.time_array[-1], stepsize)
        # plt.figure(figsize=(6.5, 4))
        # plt.plot(xs, signal, label="S")
        # plt.show()
        peaks,_ = find_peaks(signal,prominence=(0.5,None))
        rospy.loginfo(len(peaks))
        measured_time = self.time_array[-1] - self.time_array[0]
        pulse = (len(peaks)/measured_time)*60
        pulse = np.int16(pulse)
        rospy.loginfo("Pulse: "+str(pulse))
        return pulse

    def publish_pulse(self, pulse_value):
        msg_to_publish = pulse()
        msg_to_publish.pulse = pulse_value
        msg_to_publish.time.stamp = self.publish_time
        msg_to_publish.time.seq = self.seq
        self.pub.publish(msg_to_publish)
        self.seq += 1

    # Helper function to check if ROI is selected correctly
    def show_image_with_mask(self, image, forehead_mask, bottom_mask):
        bottom_dst = cv2.bitwise_and(image, bottom_mask)
        top_dst = cv2.bitwise_and(image, forehead_mask)
        dst = cv2.bitwise_or(bottom_dst, top_dst)
        cv2.imshow("Bottom", dst)
        cv2.waitKey(3)


def main():
    rospy.init_node('head_movement_listener', anonymous=False, log_level=rospy.DEBUG)
    topic = rospy.get_param("~topic", "/face_detection/mask")
    rospy.loginfo("Listening on topic '" + topic + "'")
    pulse = PulseHeadMovement(topic)
    pulse.run()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.argv = rospy.myargv()
    main()
