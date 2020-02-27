#!/usr/bin/env python
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from face_detection.msg import Mask

import cv2
import os
import rospy
import sys
import numpy as np


class FaceDetector:

    def __init__(self, topic, show_image_frame):
        self.topic = topic
        self.show_image_frame = show_image_frame
        self.bridge = CvBridge()

        self.face_publisher = rospy.Publisher("/face_detection/face", Image, queue_size=10)
        self.forehead_publisher = rospy.Publisher("/face_detection/forehead", Image, queue_size=10)
        self.bottom_face_publisher = rospy.Publisher("/face_detection/bottom_face", Image, queue_size=10)
        self.mask_publisher = rospy.Publisher("/face_detection/mask", Mask, queue_size=10)

    def run(self):
        rospy.Subscriber(self.topic, Image, self.on_image)

        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down")

    def on_image(self, data):
        try:
            # Convert ROS image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # Get gray scale image from OpenCV
        gray_scale_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Create the haar cascade
        face_cascade = cv2.CascadeClassifier(os.path.dirname(os.path.realpath(__file__)) + "/../resources/cascade.xml")

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(
            gray_scale_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Show original image, if no faces are detected
        if len(faces) is 0:
            rospy.loginfo("No faces detected!")

            if self.show_image_frame is True:
                cv2.imshow("Image", cv_image)
                cv2.waitKey(3)

            return

        # Get biggest face
        biggest_face = self.get_biggest_face(faces)
        face_x = biggest_face[0]
        face_y = biggest_face[1]
        face_w = biggest_face[2]
        face_h = biggest_face[3]

        # Crop image to biggest face
        cropped_face = cv_image[face_y: face_y + face_h, face_x: face_x + face_w]
        # Publish image to ROS
        self.publish_image(self.face_publisher, cropped_face)

        # Define region of forehead
        forehead_x = face_x + face_w / 3
        forehead_y = face_y + face_h / 16
        forehead_w = face_w / 3
        forehead_h = int(face_h / 5)

        # Crop image to forehead
        cropped_forehead = cv_image[forehead_y: forehead_y + forehead_h, forehead_x: forehead_x + forehead_w]
        # Publish image to ROS
        self.publish_image(self.forehead_publisher, cropped_forehead)

        # Define bottom region
        bottom_x = face_x + face_w / 4
        bottom_y = face_y + face_h / 2
        bottom_w = face_w / 2
        bottom_h = face_h / 2

        # Crop image to bottom region
        cropped_bottom_face = cv_image[bottom_y: bottom_y + bottom_h, bottom_x: bottom_x + bottom_w]
        # Publish image to ROS
        self.publish_image(self.bottom_face_publisher, cropped_bottom_face)

        # Publish image with mask to ROS
        forehead_mask = self.get_mask(
            gray_scale_image,
            [(forehead_x, forehead_y, forehead_w, forehead_h)])

        # Publish image with forhead mask to ROS
        bottom_mask = self.get_mask(
            gray_scale_image,
            [(bottom_x, bottom_y, bottom_w, bottom_h)]
        )
        self.publish_mask(gray_scale_image, forehead_mask, bottom_mask)

        if self.show_image_frame is True:
            # Visualize face in original image
            cv2.rectangle(cv_image, (face_x, face_y), (face_x + face_w, face_y + face_h), (255, 0, 0), 2)

            # Visualize forehead in original image
            cv2.rectangle(
                cv_image,
                (forehead_x, forehead_y),
                (forehead_x + forehead_w, forehead_y + forehead_h),
                (0, 0, 255),
                2
            )

            # Visualize bottom region in original image
            cv2.rectangle(
                cv_image,
                (bottom_x, bottom_y),
                (bottom_x + bottom_w, bottom_y + bottom_h),
                (0, 255, 0),
                2
            )

            # Show original image with visualized face and forehead
            cv2.imshow("Image", cv_image)
            cv2.waitKey(3)

    def get_biggest_face(self, faces):
        biggest_face = None

        for face in faces:
            # If width and height of rectangle is bigger, set biggest_face to current face
            if biggest_face is None or face[2] > biggest_face[2] and face[3] > biggest_face[3]:
                biggest_face = face

        return biggest_face

    def get_mask(self, gray_scale_image, rects):
        # Generate mask with zeros
        mask = np.zeros_like(gray_scale_image)

        for x, y, w, h in rects:
            # Fill in a rectangle area of the 'mask' array white
            cv2.rectangle(
                mask,
                (x, y),
                (x + w, y + h),
                (255, 255, 255),
                -1
            )

        return mask

    def publish_image(self, publisher, cv_image):
        try:
            # Convert OpenCV image back to ROS image
            ros_img = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")

            # Publish image to ROS-Topic
            publisher.publish(ros_img)
        except CvBridgeError as e:
            rospy.logerr(e)

    def publish_mask(self, cv_image, forehead_mask, bottom_face_mask):
        try:
            # Convert OpenCV image back to ROS image
            ros_img = self.bridge.cv2_to_imgmsg(cv_image, "mono8")
            ros_forehead_mask = self.bridge.cv2_to_imgmsg(forehead_mask, "mono8")
            ros_bottom_mask = self.bridge.cv2_to_imgmsg(bottom_face_mask, "mono8")
            # Create Mask ROS message from original image and roi mask
            ros_msg = Mask()
            ros_msg.image = ros_img
            ros_msg.forehead_mask = ros_forehead_mask
            ros_msg.bottom_face_mask = ros_bottom_mask

            # Publish image with mask to ROS topic
            self.mask_publisher.publish(ros_msg)
        except CvBridgeError as e:
            rospy.logerr(e)


def main(args):
    rospy.init_node('face_detection', anonymous=True, log_level=rospy.DEBUG)

    # Get ROS topic from launch parameter
    # topic = rospy.get_param("~topic", "/pylon_camera_node/image_raw")
    topic = rospy.get_param("~topic", "/webcam/image_raw")
    show_image_frame = rospy.get_param("~show_image_frame", False)
    rospy.loginfo("Listening on topic '" + topic + "'")
    rospy.loginfo("Show image frame: '" + str(show_image_frame) + "'")

    # Start face detection
    face_detector = FaceDetector(topic, show_image_frame)
    face_detector.run()

    # Destroy windows on close
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
