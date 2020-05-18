#!/usr/bin/env python
from video_input import VideoInput

import cv2
import rospy
import numpy as np


class FaceDetector:

    def __init__(self, topic, cascade_file):
        self.topic = topic
        self.cascade_file = cascade_file
        self.show_image_frame = False
        self.min_face_size = None
        self.max_face_size = None
        self.face_callback = None
        self.mask_callback = None
        self.forehead_callback = None
        self.bottom_face_callback = None

    def run(self, video_file, bdf_file, show_image_frame):
        self.show_image_frame = show_image_frame

        video_input = VideoInput(self.topic, self.cascade_file)
        video_input.frame_callback = self.detect_faces_and_extract_rois
        video_input.run(video_file, bdf_file)

    def detect_faces_and_extract_rois(self, cv_image, timestamp):
        """
        Detects the biggest face in an image and extracts multiple Region of Interests (forehead and bottom region).
        Results are published via callbacks.
        : param cv_image: The image frame that should be used for the face detection
        : param timestamp: Timestamp of the image frame
        """

        # Get gray scale image from OpenCV
        gray_scale_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Create the haar cascade
        face_cascade = cv2.CascadeClassifier(self.cascade_file)

        # Calculate the min and max face size to be detected
        # Because input frames can be of different sizes, this is calculated dynamically on the first image callback
        if not self.min_face_size or not self.max_face_size:
            height = np.size(cv_image, 0)
            self.min_face_size = (height / 3, height / 3)
            self.max_face_size = (height, height)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(
            gray_scale_image,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=self.min_face_size,
            maxSize=self.max_face_size
        )

        # Show original image, if no faces are detected
        if len(faces) is 0:
            rospy.loginfo("[FaceDetector] No faces detected!")

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
        face = cv_image[face_y: face_y + face_h, face_x: face_x + face_w]

        # Call callback with face
        if self.face_callback is not None:
            self.face_callback(face, timestamp)

        # Define region of forehead
        forehead_x = face_x + face_w / 3
        forehead_y = face_y + face_h / 16
        forehead_w = face_w / 3
        forehead_h = int(face_h / 5)

        # Crop image to forehead
        forehead = cv_image[forehead_y: forehead_y + forehead_h, forehead_x: forehead_x + forehead_w]

        # Call callback with forehead
        if self.forehead_callback is not None:
            self.forehead_callback(forehead, timestamp)

        # Define bottom region
        bottom_x = face_x + face_w / 4
        bottom_y = face_y + face_h / 2
        bottom_w = face_w / 2
        bottom_h = face_h / 2

        # Crop image to bottom region
        bottom_face = cv_image[bottom_y: bottom_y + bottom_h, bottom_x: bottom_x + bottom_w]

        # Call callback with bottom face
        if self.bottom_face_callback is not None:
            self.bottom_face_callback(bottom_face, timestamp)

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
            self.mask_callback(gray_scale_image, forehead_mask, bottom_mask, timestamp)

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
        """
        Detects the biggest face using the width and the height of the detected face rectangles.
        :param faces: Rectangles of the detected faces
        :return: The biggest face found, if one exists
        """
        if len(faces) is 1:
            return faces[0]

        biggest_face = None

        for face in faces:
            # If width and height of rectangle is bigger, set biggest_face to current face
            if biggest_face is None or face[2] > biggest_face[2] and face[3] > biggest_face[3]:
                biggest_face = face

        return biggest_face

    def get_mask(self, gray_scale_image, rects):
        """
        Returns a mask of zeros or ones with the size of the provided image.
        :param gray_scale_image: The image of which the mask should be created
        :param rects: The regions where the mask should be 1
        :return: The calculated mask out of the provided image and the rectangles
        """
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
