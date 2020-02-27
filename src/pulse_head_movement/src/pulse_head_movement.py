#!/usr/bin/env python
# -*- encoding: utf-8 -*-

__version__ = "0.1.1"

import sys
import numpy as np
import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from face_detection.msg import Mask
import common
import time

class PulseHeadMovement:

    def __init__(self, topic):
        self.topic = topic
        self.bridge = CvBridge()
        self.lk_params = dict( winSize  = (35, 35),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.points_to_track = None
        self.prev_image = None
        self.track_len = 32

    def run(self):
        rospy.Subscriber(self.topic, Mask, self.pulse_callback)
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down")

    def get_points_to_track(self, image, forehead_mask, bottom_mask):
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
        return feature_points

    # Helper function to check if ROI is selected correctly
    def show_image_with_mask(self, image, forehead_mask, bottom_mask):
        bottom_dst = cv2.bitwise_and(image, bottom_mask)
        top_dst = cv2.bitwise_and(image, forehead_mask)
        dst = cv2.bitwise_or(bottom_dst, top_dst)
        cv2.imshow("Bottom", dst)
        cv2.waitKey(3)

    def pulse_callback(self, mask):
        rospy.loginfo("Capture frame")
        try:
            # Convert ROS image to OpenCV image
            original_image = self.bridge.imgmsg_to_cv2(mask.image)
            bottom_mask = self.bridge.imgmsg_to_cv2(mask.bottom_face_mask)
            forehead_mask = self.bridge.imgmsg_to_cv2(mask.forehead_mask)
            self.show_image_with_mask(original_image,forehead_mask,bottom_mask)
            if self.points_to_track is None or len(self.points_to_track)==0:
                # get initial tracking points
                self.prev_image = original_image
                self.points_to_track = self.get_points_to_track(original_image, forehead_mask, bottom_mask)
                rospy.loginfo(self.points_to_track)
                return
            # plot tracking points
            #for new in self.points_to_track:
            #    x = new[0][0]
            #    y = new[0][1]
            #    cv2.circle(original_image, (x, y), 5, (25, 0, 0))
            #cv2.imshow("Feature", original_image)
            #cv2.waitKey(3)

            # track points with lucas kanade tracker
            t0 = time.time()
            t1, t2, t3 = 0, 0, 0
            # make a copy for visualization
            vis = original_image.copy()
            if len(self.points_to_track) > 0:
                img0, img1 = self.prev_image, original_image
                # get tracking points
                t1 = time.time() - t0
                p0 = np.float32([tr[-1] for tr in self.points_to_track]).reshape(-1, 1, 2)
                # apply Lucas-Kanade tracker with optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
                t2 = time.time() - t0
                d = abs(p0 - p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.points_to_track, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    np.append(tr,(x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)

                t3 = time.time() - t0
                self.points_to_track = new_tracks
                cv2.polylines(vis, [np.int32(tr) for tr in self.points_to_track], False, (0, 255, 0))
                cv2.imshow('lk_track', vis)
                cv2.waitKey(3)
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
