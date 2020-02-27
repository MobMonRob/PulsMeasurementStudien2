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
        self.pulse_processor.buffer_size = 30 * 5  # MAX_FPS * 5

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        cv_image = cv2.flip(cv_image, 1)
        frame = np.copy(cv_image)
        self.pulse_processor.run(frame)

        if len(self.pulse_processor.data_buffer) == self.pulse_processor.buffer_size:
            rospy.loginfo("BPM: " + str(self.pulse_processor.bpm))


def main(args):
    rospy.init_node('face_detection', anonymous=True, log_level=rospy.DEBUG)

    topic = rospy.get_param("~topic", "/face_detection/face")
    rospy.loginfo("Listening on topic '" + topic + "'")

    image_converter = ImageConverter(topic)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
