#!/usr/bin/env python
from __future__ import print_function
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from pulse_measure import PulseMeasurement

import cv2
import rospy
import sys


class ImageConverter:

    def __init__(self, topic):
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.bridge = CvBridge()
        self.pulse_processor = PulseMeasurement()

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        self.pulse_processor.run(cv_image)

def main(args):
    rospy.init_node('face_detection', anonymous=True, log_level=rospy.DEBUG)

    topic = rospy.get_param("~topic", "/face_detection/forehead")
    rospy.loginfo("Listening on topic '" + topic + "'")

    image_converter = ImageConverter(topic)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)