#!/usr/bin/env python
from __future__ import print_function
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

import cv2
import os
import rospy
import sys


class ImageConverter:

    def __init__(self, topic):
        self.image_pub = rospy.Publisher("/face_detection/image_raw", Image, queue_size=10)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Create the haar cascade
        face_cascade = cv2.CascadeClassifier(os.path.dirname(os.path.realpath(__file__)) + "/../resources/cascade.xml")

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) is 0:
            rospy.loginfo("No faces detected!")
            return

        face = faces[0]
        x = face[0]
        y = face[1]
        w = face[2]
        h = face[3]

        cropped_image = cv_image[y: y + (int(h / 3.2)), x + (w / 4): x + (w / 4) + (w / 2)]

        cv2.imshow("Cropped", cropped_image)
        cv2.waitKey(3)

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cropped_image, "bgr8"))
        except CvBridgeError as e:
            print(e)


def main(args):
    rospy.init_node('face_detection', anonymous=True, log_level=rospy.DEBUG)

    # topic = rospy.get_param("~topic", "/pylon_camera_node/image_raw")
    topic = rospy.get_param("~topic", "/webcam/image_raw")
    rospy.loginfo("Listening on topic '" + topic + "'")

    image_converter = ImageConverter(topic)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
