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
            cv2.imshow("Image", cv_image)
            cv2.waitKey(3)
            return

        biggest_face = self.get_biggest_face(faces)
        face_x = biggest_face[0]
        face_y = biggest_face[1]
        face_w = biggest_face[2]
        face_h = biggest_face[3]
        cv2.rectangle(cv_image, (face_x, face_y), (face_x + face_w, face_y + face_h), (255, 0, 0), 2)

        forehead_x = face_x + face_w / 4
        forehead_y = face_y
        forehead_w = face_w / 2
        forehead_h = int(face_h / 3.2)

        cropped_image = cv_image[forehead_y: forehead_y + forehead_h, forehead_x: forehead_x + forehead_w]
        cv2.rectangle(
            cv_image,
            (forehead_x, forehead_y),
            (forehead_x + forehead_w, forehead_y + forehead_h),
            (0, 0, 255),
            2
        )

        cv2.imshow("Image", cv_image)
        cv2.waitKey(3)

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cropped_image, "bgr8"))
        except CvBridgeError as e:
            print(e).0

    def get_biggest_face(self, faces):
        biggest_face = None

        for face in faces:
            if biggest_face is None or face[2] > biggest_face[2] and face[3] > biggest_face[3]:
                biggest_face = face

        return biggest_face


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
