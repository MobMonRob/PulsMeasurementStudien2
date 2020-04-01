#!/usr/bin/env python
from __future__ import print_function
from pulse_measure import PulseMeasurement
from face_detection import FaceDetector

import cv2
import rospy
import sys


class ImageConverter:

    def __init__(self):
        self.pulse_processor = PulseMeasurement()

    def run(self, topic, cascade_file, show_image_frame):
        # Start face detection
        face_detector = FaceDetector(topic, cascade_file, show_image_frame)
        face_detector.face_callback = self.pulse_processor.run
        face_detector.run()

        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down")


def main(args):
    rospy.init_node('face_detection', anonymous=True, log_level=rospy.DEBUG)

    # Get ROS topic from launch parameter
    topic = rospy.get_param("~topic", "/webcam/image_raw")
    rospy.loginfo("Listening on topic '" + topic + "'")

    cascade_file = rospy.get_param("~cascade_file", "")
    rospy.loginfo("Show cascade_file frame: '" + str(cascade_file) + "'")

    show_image_frame = rospy.get_param("~show_image_frame", False)
    rospy.loginfo("Show image frame: '" + str(show_image_frame) + "'")

    image_converter = ImageConverter()
    image_converter.run(topic, cascade_file, show_image_frame)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)