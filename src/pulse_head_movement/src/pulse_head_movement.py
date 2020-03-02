#!/usr/bin/env python
# -*- encoding: utf-8 -*-

__version__ = "0.1.1"

import sys
import numpy as np
import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from face_detection.msg import Mask
import time

class PulseHeadMovement:

    def __init__(self, topic):
        self.topic = topic
        self.bridge = CvBridge()
        self.lk_params = dict( winSize  = (35, 35),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.03))
        self.points_to_track = None
        self.prev_image = None
        self.track_len = 32
        self.refresh_rate = 50
        self.frame_index = 0
        self.y_tracking_signal = None
        self.start_time = None
        self.refresh_time = None

    def run(self):
        rospy.Subscriber(self.topic, Mask, self.pulse_callback)
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down")

    def get_points_to_track(self, image, forehead_mask, bottom_mask):
        image = cv2.equalizeHist(image)
        # get the tracking points in the bottom face region
        # parameter for feature points in bottom face region. As there are some feature rich points, the quality level
        # is low to include more points
        bottom_feature_params = dict(maxCorners=100, qualityLevel=0.05, minDistance=7, blockSize=7)
        bottom_points = cv2.goodFeaturesToTrack(image, mask=bottom_mask, **bottom_feature_params)
        feature_points = np.array(bottom_points, dtype=np.float32)
        # get the tracking points in the forehead region
        # parameter for feature points in forehead region. As most points are not very feature rich, the quality level
        # is high to enable better tracking with lk tracker
        forehead_feature_params = dict(maxCorners=100, qualityLevel=0.6, minDistance=7, blockSize=7)
        forehead_points = cv2.goodFeaturesToTrack(image, mask=forehead_mask, **forehead_feature_params)
        forehead_points = np.array(forehead_points, dtype=np.float32)
        # put tracking points of both regions in one array and return feature points
        feature_points = np.append(feature_points, forehead_points, axis=0)
        self.y_tracking_signal = np.empty([len(feature_points), self.refresh_rate-1], dtype=np.float32)
        return feature_points

    def calculate_optical_flow(self, image):
        # track points with lucas kanade tracker
        # make a copy for visualization
        vis = image.copy()
        if len(self.points_to_track) > 0:
            img0, img1 = cv2.equalizeHist(self.prev_image), cv2.equalizeHist(image)
            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, self.points_to_track, None, **self.lk_params)
            point_index = 0
            for p in p1:
                cv2.circle(vis, (p[0][0], p[0][1]), 2, (0, 255, 0), -1)
                self.y_tracking_signal[point_index][(self.frame_index%self.refresh_rate)-1] = p[0][1]
                point_index += 1

            self.points_to_track = p1
            #rospy.loginfo(self.frame_index%self.refresh_rate)
            cv2.imshow('lk_track', vis)
            cv2.waitKey(3)

    def process_saved_points(self):
        fps = self.calculate_fps()
        rospy.loginfo("FPS: " + str(fps))
        self.interpolate_points()
        self.apply_butterworth_filter()
        self.process_PCA()
        self.find_most_periodic_signal()
        self.calculate_pulse()
        return

    def calculate_fps(self):
        timespan = self.refresh_time-self.start_time
        time_sec = timespan.to_sec()
        fps = self.refresh_rate/time_sec
        return fps

    def interpolate_points(self):
        return

    def apply_butterworth_filter(self):
        return

    def process_PCA(self):
        return

    def find_most_periodic_signal(self):
        return

    def calculate_pulse(self):
        return

    # Helper function to check if ROI is selected correctly
    def show_image_with_mask(self, image, forehead_mask, bottom_mask):
        bottom_dst = cv2.bitwise_and(image, bottom_mask)
        top_dst = cv2.bitwise_and(image, forehead_mask)
        dst = cv2.bitwise_or(bottom_dst, top_dst)
        cv2.imshow("Bottom", dst)
        cv2.waitKey(3)

    def pulse_callback(self, mask):
        self.frame_index = mask.time.seq
        rospy.loginfo("Capture frame: "+str(self.frame_index))
        try:
            # Convert ROS image to OpenCV image
            original_image = self.bridge.imgmsg_to_cv2(mask.image)
            bottom_mask = self.bridge.imgmsg_to_cv2(mask.bottom_face_mask)
            forehead_mask = self.bridge.imgmsg_to_cv2(mask.forehead_mask)
            # self.show_image_with_mask(original_image,forehead_mask,bottom_mask)
            # refresh the points to track after a certain frame rate
            if self.frame_index % self.refresh_rate == 0 \
                    or self.points_to_track is None \
                    or len(self.points_to_track) == 0:
                if self.y_tracking_signal is not None:
                    # We already stored some y points from which we want to calculate the pulse
                    rospy.loginfo(self.y_tracking_signal)
                    self.refresh_time = mask.time.stamp
                    self.process_saved_points()
                # get initial tracking points
                self.prev_image = original_image
                self.start_time = mask.time.stamp
                self.points_to_track = self.get_points_to_track(original_image, forehead_mask, bottom_mask)
                # rospy.loginfo(self.points_to_track)
                return
            if self.points_to_track is not None:
                self.calculate_optical_flow(original_image)
            self.prev_image = original_image

        except CvBridgeError as e:
            rospy.logerr(e)
            return


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
