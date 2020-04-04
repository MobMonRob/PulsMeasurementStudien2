#!/usr/bin/env python
<<<<<<< HEAD
=======
from bdf_processor import BdfProcessor
>>>>>>> master
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

import cv2
import rospy
import numpy as np
import time


class FaceDetector:

    def __init__(self, topic, cascade_file, show_image_frame):
        self.topic = topic
        self.cascade_file = cascade_file
        self.show_image_frame = show_image_frame
        self.bridge = CvBridge()
        self.image_sequence = 0
        self.start = 0
        self.frames = 0
        self.count = 0
        self.biggest_face = None
        self.min_size = None
        self.max_size = None
        self.face_callback = None
        self.mask_callback = None
        self.forehead_callback = None
        self.bottom_face_callback = None

    def run(self, bdf_file):
        self.start = time.time()

        # Start bdf processor
        if bdf_file and bdf_file != "None":
            bdf_processor = BdfProcessor(bdf_file)
            bdf_processor.run()
        rospy.Subscriber(self.topic, Image, self.on_image)

    def on_image(self, data, convert=True):
        recalculate = (self.count % 1) == 0
        self.count += 1
        self.frames += 1

        if self.frames is 60:
            end = time.time()
            seconds = end - self.start
            rospy.loginfo("Time taken: " + str(seconds) + " seconds")

            fps = 60 / seconds
            rospy.loginfo("Estimated frames per second: " + str(fps))

            self.start = time.time()
            self.frames = 0

        if convert:
            try:
                # Convert ROS image to OpenCV image
                cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                rospy.logerr(e)
                return
        else:
            cv_image = data

        # Get gray scale image from OpenCV
        gray_scale_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        if recalculate:
            # Create the haar cascade
            face_cascade = cv2.CascadeClassifier(self.cascade_file)

            if not self.min_size or not self.max_size:
                height = np.size(cv_image, 0)
                self.min_size = (height / 3, height / 3)
                self.max_size = (height, height)

            # Detect faces in the image
            faces = face_cascade.detectMultiScale(
                gray_scale_image,
                scaleFactor=1.15,
                minNeighbors=5,
                minSize=self.min_size,
                maxSize=self.max_size
            )

            # Show original image, if no faces are detected
            if len(faces) is 0:
                rospy.loginfo("No faces detected!")
                self.count = 0

                if self.show_image_frame is True:
                    cv2.imshow("Image", cv_image)
                    cv2.waitKey(3)
                return

            # Get biggest face
            self.biggest_face = self.get_biggest_face(faces)

        face_x = self.biggest_face[0]
        face_y = self.biggest_face[1]
        face_w = self.biggest_face[2]
        face_h = self.biggest_face[3]

        # Crop image to biggest face
        face = cv_image[face_y: face_y + face_h, face_x: face_x + face_w]

        # Call callback with face
        if self.face_callback is not None:
            self.face_callback(face)

        # Define region of forehead
        forehead_x = face_x + face_w / 3
        forehead_y = face_y + face_h / 16
        forehead_w = face_w / 3
        forehead_h = int(face_h / 5)

        # Crop image to forehead
        forehead = cv_image[forehead_y: forehead_y + forehead_h, forehead_x: forehead_x + forehead_w]

        # Call callback with forehead
        if self.forehead_callback is not None:
            self.forehead_callback(forehead)

        # Define bottom region
        bottom_x = face_x + face_w / 4
        bottom_y = face_y + face_h / 2
        bottom_w = face_w / 2
        bottom_h = face_h / 2

        # Crop image to bottom region
        bottom_face = cv_image[bottom_y: bottom_y + bottom_h, bottom_x: bottom_x + bottom_w]

        # Call callback with bottom face
        if self.bottom_face_callback is not None:
            self.bottom_face_callback(bottom_face)

        # Get forehead mask
        forehead_mask = self.get_mask(
            gray_scale_image,
            [(forehead_x, forehead_y, forehead_w, forehead_h)]
        )

        # Get bottom face mask
        bottom_mask = self.get_mask(
            gray_scale_image,
            [(bottom_x, bottom_y, bottom_w, bottom_h)]
        )

        # Call callback with mask
        if self.mask_callback is not None:
            self.mask_callback(gray_scale_image, forehead_mask, bottom_mask, rospy.Time.now())

        self.image_sequence += 1

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
        if len(faces) is 1:
            return faces[0]

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
