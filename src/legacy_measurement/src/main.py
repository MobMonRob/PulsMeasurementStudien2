#!/usr/bin/env python
from __future__ import print_function
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from pulse_measure import PulseMeasurement
from face_detection import FaceDetector

import cv2
import rospy
import sys
import numpy as np


class ImageConverter:

    def __init__(self, topic):
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.bridge = CvBridge()
        self.pulse_processor = PulseMeasurement()
        self.pulse_processor.buffer_size = 1000

    def callback(self, cv_image, time):
        cv_image = cv2.flip(cv_image, 1)
        frame = np.copy(cv_image)
        self.pulse_processor.run(frame, time)


def main():
    rospy.init_node('legacy_measurement', anonymous=False, log_level=rospy.DEBUG)

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

    # Start heart rate measurement
    image_converter = ImageConverter(topic)

    face_detector = FaceDetector(topic, cascade_file)
    face_detector.bottom_face_callback = image_converter.callback
    face_detector.run(video_file, bdf_file, show_image_frame)

    rospy.spin()
    rospy.loginfo("Shutting down")


if __name__ == '__main__':
    sys.argv = rospy.myargv()
    main()
