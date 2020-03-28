#!/usr/bin/env python
# -*- encoding: utf-8 -*-

__version__ = "0.1.1"

import sys
import numpy as np
import cv2
import rospy
import peakutils
from pulse_chest_strap.msg import pulse
from scipy import stats
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
    """
    Butter bandpass filter.
    Inspired from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


class PulseHeadMovement:

    def __init__(self, topic):
        """
        Constructor.
        :param topic: the topic to listen to (video input stream)
        """
        self.topic = topic
        # bridge to convert ROS image to OpenCV image
        self.bridge = CvBridge()
        # set up ROS publisher
        self.pub = rospy.Publisher('head_movement_pulse', pulse, queue_size=10)
        # sequence of published pulse values, published with each pulse message
        self.published_pulse_value_sequence = 0
        # previous image is needed for lucas kanade optical flow tracker (see calculate_optical_flow method)
        self.previous_image = None
        # the refresh rate specifies how many frames each calculated feature points should be tracked
        self.refresh_rate = 300
        # the publish rate specifies how many frames are between each published pulse value
        self.publish_rate = 50
        # helper variable to measure performance
        self.fps = 0
        self.frame_index = 0
        # current positions of the tracking point for each buffer position
        # three dimensional array of the form
        # [[[x1_b1,y1_b1], [x2_b1,y2_b1],...],
        #  [[x1_b2,y1_b2], [x2_b2,y2_b2],...],
        #   ...]
        #   for each point
        self.buffer_points = []
        # time for each buffer meaning the times associated with the frames of this buffer
        # two dimensional array of the form
        # [[t1_b1,t2_b1,t3_b1,...]
        #  [t1_b2,t2_b2,t3_b2,...],
        #  ...]
        self.buffered_time_arrays = []
        # y positions over the time for each tracking point for each buffer position
        # three dimensional array of the form
        # [[[y1_t1_b1,y1_t2_b1,y1_t3_b1,...],[y2_t1_b1,y2_t2_b1,y2_t3_b1,...],...]
        #   [y1_t1_b2,y1_t2_b2,y1_t3_b2,...],[y2_t1_b2,y2_t2_b2,y2_t3_b2,...],...],
        #   ...]
        self.buffered_y_tracking_signal = []

    def run(self):
        """
        Method to subscribe to imaging topic.
        Whenever a image is published to the topic, the pulse_callback method is published.
        """
        rospy.Subscriber(self.topic, Mask, self.pulse_callback)
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down")

    def pulse_callback(self, mask):
        """
        Callback method for incoming video frames
        :param mask: the message published to the topic (contains gray-image, mask from bottom part of the face
                                                         and mask for forehead)
        """
        rospy.loginfo("Capture frame: " + str(self.frame_index))
        try:
            original_image, bottom_mask, forehead_mask = self.convert_ros_images_to_opencv(mask)
            self.get_current_tracking_points_position(original_image, mask.time.stamp.to_sec())
            if self.frame_index % self.publish_rate == 0:
                self.add_new_points_to_buffer(original_image, forehead_mask, bottom_mask, mask.time.stamp)
            self.frame_index += 1
            self.previous_image = original_image
        except CvBridgeError as e:
            rospy.logerr(e)
            return

    def convert_ros_images_to_opencv(self, mask):
        """
        Convert the incoming ros images to OpenCV images for further processing
        :param mask: the message published to the topic (contains gray-image, mask from bottom part of the face
                                                         and mask for forehead)
        """
        original_image = self.bridge.imgmsg_to_cv2(mask.image)
        bottom_mask = self.bridge.imgmsg_to_cv2(mask.bottom_face_mask)
        forehead_mask = self.bridge.imgmsg_to_cv2(mask.forehead_mask)
        return original_image, bottom_mask, forehead_mask

    def get_current_tracking_points_position(self, current_image, time):
        """
        Get the current position for each tracking point in each buffer position
        :param current_image: the current image to calculate the optical flow on
        :param time: the corresponding time for the image frame
        """
        buffer_index = 0
        for points in self.buffer_points:
            # calculate new tracking points position for each buffer position
            new_points = self.calculate_optical_flow(current_image, points)
            # replace old [x,y] positions of tracking points in buffer_points array
            self.buffer_points[buffer_index] = new_points
            # time position of each point meaning the sequential position of the points for each buffer position
            time_position = ((self.frame_index % self.publish_rate) - 1) + buffer_index * self.publish_rate
            # add y point in the time movement for the points
            tracking_point_index = 0
            for point in new_points:
                self.buffered_y_tracking_signal[buffer_index][tracking_point_index][time_position] = point[0][1]
                tracking_point_index += 1
            self.buffered_time_arrays[buffer_index][time_position] = time
            buffer_index += 1

    def calculate_optical_flow(self, image, previous_points):
        """
        Calculate the optical flow (track the points) with the lucas kanade tracker
        See https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html for more information on optical flow
        :param image: the current image to calculate difference to previous image
        :param previous_points: the previous position of the points
        """
        # make a copy for visualization
        vis = image.copy()
        img0, img1 = cv2.equalizeHist(self.previous_image), cv2.equalizeHist(image)
        lk_params = dict(winSize=(35, 35),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.03))
        new_points, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, previous_points, None, **lk_params)
        point_index = 0
        # visualization of tracking points
        for p in new_points:
            cv2.circle(vis, (p[0][0], p[0][1]), 2, (0, 255, 0), -1)
            point_index += 1
        cv2.imshow('lk_track', vis)
        cv2.waitKey(3)
        return new_points

    def add_new_points_to_buffer(self, current_image, forehead_mask, bottom_mask, timestamp):
        """
        Add new points to the buffer in the frequency of self.publish_rate and add them to the buffer
        """
        current_points_to_track = self.get_points_to_track(current_image, forehead_mask, bottom_mask)
        self.edit_buffer(current_points_to_track, timestamp)

    def get_points_to_track(self, image, forehead_mask, bottom_mask):
        """
        get new tracking points using OpenCV goodFeaturesToTrack.
        New tracking points are added in the frequency of self.publish_rate
        For more information on the goodFeaturesToTrack method see
        https://docs.opencv.org/master/d4/d8c/tutorial_py_shi_tomasi.html and
        https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html
        """
        # histogram of the image is equalized to increase contrast
        image = cv2.equalizeHist(image)
        # get the tracking points in the bottom face region
        bottom_feature_params = dict(maxCorners=100, qualityLevel=0.01, minDistance=7, blockSize=7)
        bottom_points = cv2.goodFeaturesToTrack(image, mask=bottom_mask, **bottom_feature_params)
        feature_points = np.array(bottom_points, dtype=np.float32)
        # get the tracking points in the forehead region
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
        return feature_points

    def edit_buffer(self, current_points_to_track, publish_time):
        """
        If new points are added, the buffer has to be edited.
        If the buffer is full, points in the last position are removed from the array and are further processed
        to calculate pulse.
        The new points are added in the front of the buffer.
        :param current_points_to_track: the new points to add. [x,y] coordinates are pushed into self.buffer_points.
        :param publish_time: the timestamp for the ros message to publish the pulse value
        """
        if len(self.buffer_points) < self.refresh_rate/self.publish_rate:
            # as the buffer is not full, there are no points yet to process.
            rospy.loginfo("array not full yet ")
        else:
            # process points on the last array position
            self.buffer_points.pop()
            points_calculate_pulse = self.buffered_y_tracking_signal.pop()
            current_time_array = self.buffered_time_arrays.pop()
            self.process_saved_points(points_calculate_pulse, current_time_array, publish_time)
        # push new points in buffer
        self.buffer_points.insert(0, current_points_to_track)
        # initialize empty array for the y tracking signal of the new points and the time
        # and push it to self.buffered_y_tracking_signal and self.buffered_time_array
        self.buffered_y_tracking_signal.insert(0, np.empty([len(current_points_to_track), self.refresh_rate-1],
                                                      dtype=np.float32))
        self.buffered_time_arrays.insert(0, np.empty(self.refresh_rate-1))
        return

    def process_saved_points(self, y_tracking_signal, time_array, publish_time):
        """
        central method for processing the points, to calculate pulse in the end.
        :param y_tracking_signal: the tracking signal to calculate pulse on
        :param time_array: the time of the single positions of the points tracked
        :param publish_time: the timestamp for the ros message to publish the pulse value
        """
        self.calculate_fps(time_array)
        rospy.loginfo("FPS: " + str(self.fps))
        stable_signal = self.remove_erratic_trajectories(y_tracking_signal)
        interpolated_points = self.interpolate_points(stable_signal, time_array)
        filtered_signal = self.apply_butterworth_filter(interpolated_points, time_array)
        pca_array = self.process_PCA(filtered_signal, time_array)
        signal = self.find_most_periodic_signal(pca_array, time_array)
        pulse = self.calculate_pulse(signal, time_array)
        self.publish_pulse(pulse, publish_time)
        return

    def calculate_fps(self, time_array):
        """
        Helper method to calculate fps for performance measuring.
        """
        timespan = time_array[-1]-time_array[0]
        rospy.loginfo("Measured timespan: "+str(timespan))
        self.fps = self.refresh_rate/timespan

    def remove_erratic_trajectories(self, y_tracking_signal):
        """
        Some feature points behave unstable. This method outliers.
        """
        stable_signal = []
        rounded_signal = np.rint(y_tracking_signal)
        y_point_distance = np.diff(rounded_signal)
        y_point_distance = np.absolute(y_point_distance)
        max_distances = []
        # get the maximum distance for each point
        for point in y_point_distance:
            max_distance = np.amax(point)
            max_distances.append(max_distance)
        mode,_ = stats.mode(max_distances)
        mode = mode[0]
        #only keep the points with max lower than the mode
        point_index = 0
        for point in y_tracking_signal:
            if max_distances[point_index] <= mode:
                stable_signal.append(point)
            point_index += 1
        stable_signal = np.array(stable_signal)
        # uncomment the following lines to see the maximum distaces each point has moved. For debugging.
        # xs = np.arange(len(max_distances))
        # filename = "/home/studienarbeit/Dokumente/" + str(self.seq) + "max_distance"
        # plt.figure(figsize=(6.5, 4))
        # plt.plot(xs, max_distances, label="S")
        # plt.savefig(filename)
        return stable_signal

    def interpolate_points(self, y_tracking_signal, time_array):
        """
        As the movement of the points is measured with max. 30 fps (normally lower) we interpolate the signal to
        250Hz which is the normal sample rate of an ECG.
        For this, cubic spline interpolation is applied.
        :param y_tracking_signal: the signal resulting from remove_erratic_trajectories
        :param time_array:
        """
        sample_rate = 250
        stepsize = 1./sample_rate
        xs = np.arange(time_array[0], time_array[-1],stepsize)
        interpolated_points = np.empty([np.size(y_tracking_signal,0), np.size(xs)])
        point_index = 0
        for row in y_tracking_signal:
            cs = interpolate.interp1d(time_array, row, kind="cubic",copy=False,axis=0)
            array_interpolated = cs(xs)
            interpolated_point_index = 0
            for point in array_interpolated:
                interpolated_points[point_index][interpolated_point_index] = point
                interpolated_point_index += 1
            point_index+=1
        # uncomment the following lines to see the interpolated signal
        # for a random point (i.e. at position 8). For debugging.
        # filename = "/home/studienarbeit/Dokumente/" + str(self.seq) + "_move_signal_"
        # plt.figure(figsize=(6.5, 4))
        # plt.plot(xs, interpolated_points[8], label="S")
        # plt.savefig(filename)
        return interpolated_points

    def apply_butterworth_filter(self, input_signal, time_array):
        """
        Remove the movements which are not in the frequency of the pulse.
        :param input_signal: the signal resulting from interpolate_points
        :param time_array:
        """
        sample_rate = len(input_signal[0])/(time_array[-1]-time_array[0])
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
                filtered_point_index += 1
            point_index += 1
        # uncomment the following lines to see the interpolated signal
        # for a random point (i.e. at position 6). For debugging.
        # np.savetxt("/home/studienarbeit/Dokumente/y_points_filtered.csv", filtered_signal, delimiter=";")
        # stepsize = 1. / sample_rate
        # xs = np.arange(self.time_array[0], self.time_array[-1], stepsize)
        # plt.figure(figsize=(6.5, 4))
        # plt.plot(xs, filtered_signal[6], label="S")
        # plt.show()
        return filtered_signal

    def process_PCA(self, filtered_signal, time_array):
        """
        process PCA to get the 5 main movement directions of the signal.
        :param filtered_signal: the signal resulting from apply_butterwort_filter
        :param time_array:
        """
        filtered_signal = filtered_signal.transpose()
        pca = PCA(n_components=5)
        pca_array=pca.fit_transform(filtered_signal)
        pca_array = pca_array.transpose()
        # uncomment the following lines to see the signal after PCA. For debugging.
        # filename = "/home/studienarbeit/Dokumente/" + str(self.seq) + "_pca_signal_"
        # sample_rate = len(filtered_signal[0]) / (time_array[-1] - time_array[0])
        # stepsize = 1. / sample_rate
        # xs = np.arange(time_array[0], time_array[-1], stepsize)
        # plt.figure(figsize=(6.5, 4))
        # for row in pca_array:
        #     plt.plot(xs, row, label="S")
        # plt.savefig(filename)
        return pca_array

    def find_most_periodic_signal(self, pca_array, time_array):
        """
        Find the most periodic signal to use for pulse calculation.
        Uses the highest value of the autocorrelation funciton
        :param pca_array: the array resulting from process_PCA
        :param time_array:
        """
        highest_correlation = 0
        best_signal = None
        for signal in pca_array:
            series = pd.Series(signal)
            autocorrelation = series.autocorr()
            if autocorrelation > highest_correlation:
                best_signal = signal
        # uncomment the following lines to see the most periodic signal of the PCA. For debugging.
        # filename = "/home/studienarbeit/Dokumente/" + str(self.seq) + "_periodic_pca_signal_"
        # sample_rate = len(pca_array[0]) / (time_array[-1] - time_array[0])
        # stepsize = 1. / sample_rate
        # pca_array = None
        # xs = np.arange(time_array[0], time_array[-1], stepsize)
        # plt.figure(figsize=(6.5, 4))
        # plt.plot(xs, best_signal, label="S")
        # plt.show()
        return best_signal

    def calculate_pulse(self, signal, time_array):
        """
        calculates the pulse value from the processed signal.
        Uses automatic peak detection and divides number of peaks through measured
        this is multiplied by 60 to get bpm.
        :param signal: the finally processed signal
        """
        peaks,_ = find_peaks(signal, prominence=(0.3,None))
        measured_time = time_array[-1] - time_array[0]
        pulse = (len(peaks) / measured_time) * 60
        pulse = np.int16(pulse)
        rospy.loginfo("Pulse: " + str(pulse))
        # uncomment the following lines to see the final signal with the detected peaks. For debugging.
        # sample_rate = len(signal) / (time_array[-1] - time_array[0])
        # stepsize = 1. / sample_rate
        # xs = np.arange(time_array[0], time_array[-1], stepsize)
        # filename = "/home/studienarbeit/Dokumente/" + str(self.seq) + "_final_signal_"
        # plt.figure(figsize=(6.5, 4))
        # plt.plot(xs, signal, label="S")
        # plt.plot(xs[peaks], signal[peaks], "x")
        # plt.savefig(filename)
        return pulse

    def publish_pulse(self, pulse_value, publish_time):
        """
        Publish the calculated pulse to ROS. Message is of type pulse() from pulse_chest_strap package.
        :param pulse_value: the calculated pulse value
        :param publish_time: the timestamp of the last incoming frame
        """
        msg_to_publish = pulse()
        msg_to_publish.pulse = pulse_value
        msg_to_publish.time.stamp = publish_time
        msg_to_publish.time.seq = self.published_pulse_value_sequence
        self.pub.publish(msg_to_publish)
        self.published_pulse_value_sequence += 1

    def show_image_with_mask(self, image, forehead_mask, bottom_mask):
        """
        Helper function to check if ROI is selected correctly
        """
        bottom_dst = cv2.bitwise_and(image, bottom_mask)
        top_dst = cv2.bitwise_and(image, forehead_mask)
        dst = cv2.bitwise_or(bottom_dst, top_dst)
        cv2.imshow("Bottom", dst)
        cv2.waitKey(3)


def main():
    """
    Main.
    Get topic to listen to from launch file and starts main loop in with pulse.run().
    """
    rospy.init_node('head_movement_listener', anonymous=False, log_level=rospy.DEBUG)
    topic = rospy.get_param("~topic", "/face_detection/mask")
    rospy.loginfo("Listening on topic '" + topic + "'")
    pulse = PulseHeadMovement(topic)
    pulse.run()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.argv = rospy.myargv()
    main()